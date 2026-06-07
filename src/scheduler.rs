//! Continuous batching with iteration-level scheduling.
//!
//! ## What "continuous" means
//!
//! Static batching groups requests, runs them all to completion, and only then
//! admits the next group. A short request stuck in a batch with a long one
//! wastes the slot the moment it finishes. Continuous batching instead makes a
//! scheduling decision *every decode iteration*. A sequence that emits its eos
//! token leaves the batch immediately and a waiting request takes its place on
//! the very next step. The GPU stays saturated and tail latency drops.
//!
//! ## What this scheduler does each step
//!
//! 1. **Admission.** While there is room in the batch and free KV blocks, pull
//!    waiting requests into the running set and reserve their prompt blocks.
//! 2. **Decode.** Every running sequence advances by one token. Before it does,
//!    it must be able to reserve a block for that token.
//! 3. **Preemption.** If the cache cannot grow every running sequence, the
//!    scheduler preempts the newest (least progressed) sequences, returning
//!    their blocks, until the rest fit. Preempted sequences go back to the
//!    waiting queue and resume later. This is the recompute-based preemption
//!    strategy: simple, and it never deadlocks.
//! 4. **Retirement.** Sequences that hit eos or their length limit are removed
//!    and their blocks freed.
//!
//! The scheduler is deliberately decoupled from the model. It decides *which*
//! sequences run; the engine performs the forward pass. That separation is what
//! makes the batching policy unit-testable without a GPU.

use crate::model::TokenId;
use crate::paged_cache::{PagedKVCache, SeqId};
use std::collections::VecDeque;

/// The lifecycle state of a sequence inside the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// In the queue, not yet admitted to the running batch.
    Waiting,
    /// In the running batch, advancing one token per step.
    Running,
    /// Was running, got preempted, waiting to resume.
    Preempted,
    /// Reached eos or its length limit.
    Finished,
}

/// A request the scheduler tracks from admission to completion.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    /// Tokens generated so far. Does not include the prompt.
    pub output: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub state: SeqState,
}

impl Sequence {
    pub fn new(id: SeqId, prompt: Vec<TokenId>, max_new_tokens: usize) -> Self {
        Self {
            id,
            prompt,
            output: Vec::new(),
            max_new_tokens,
            state: SeqState::Waiting,
        }
    }

    /// Total tokens currently stored for this sequence: prompt plus output.
    pub fn total_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    /// Whether the sequence has produced all the tokens it was asked for.
    pub fn is_complete(&self) -> bool {
        self.output.len() >= self.max_new_tokens
    }
}

/// The decisions the scheduler made on one iteration. The engine consumes this
/// to know which forward passes to run, and tests assert on it directly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StepPlan {
    /// Sequences admitted from the waiting queue this step (their prompts are
    /// to be prefilled).
    pub admitted: Vec<SeqId>,
    /// Sequences that will decode one token this step, in batch order.
    pub decode_batch: Vec<SeqId>,
    /// Sequences preempted this step to free KV blocks.
    pub preempted: Vec<SeqId>,
    /// Sequences retired this step (finished).
    pub finished: Vec<SeqId>,
}

impl StepPlan {
    pub fn is_idle(&self) -> bool {
        self.admitted.is_empty()
            && self.decode_batch.is_empty()
            && self.preempted.is_empty()
            && self.finished.is_empty()
    }
}

/// Limits that shape the scheduler's decisions.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences decoding concurrently.
    pub max_batch_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { max_batch_size: 8 }
    }
}

/// The continuous batching scheduler. It owns the KV-cache and the set of live
/// sequences, and exposes one method, [`Scheduler::schedule`], that plans a
/// single iteration.
pub struct Scheduler {
    config: SchedulerConfig,
    cache: PagedKVCache,
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, cache: PagedKVCache) -> Self {
        Self {
            config,
            cache,
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    pub fn cache(&self) -> &PagedKVCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut PagedKVCache {
        &mut self.cache
    }

    /// Submit a request. It joins the back of the waiting queue.
    pub fn submit(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Are there any sequences left to work on?
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Look up a running sequence by id, for the engine to read its context and
    /// append a generated token.
    pub fn running_seq_mut(&mut self, id: SeqId) -> Option<&mut Sequence> {
        self.running.iter_mut().find(|s| s.id == id)
    }

    /// Plan one scheduling iteration. This mutates the cache (reserving and
    /// freeing blocks) and the waiting and running sets, and returns the plan
    /// the engine should execute. It does *not* run any forward pass; that is
    /// the engine's job. Splitting it this way is what makes the policy
    /// testable in isolation.
    pub fn schedule(&mut self) -> StepPlan {
        let mut plan = StepPlan::default();

        // Step 1: admission. Pull waiting requests in while the batch has room
        // and the cache can hold their prompts.
        while self.running.len() < self.config.max_batch_size {
            let next = match self.waiting.front() {
                Some(s) => s,
                None => break,
            };
            // A preempted sequence keeps its output and must re-reserve blocks
            // for prompt + output. A fresh one only needs its prompt.
            let tokens_to_place = next.total_len();
            // Reserve via a temporary admit so blocks_needed_for sees an empty
            // table. We admit then append, rolling back on failure.
            let id = next.id;
            self.cache.admit(id);
            match self.cache.append(id, tokens_to_place) {
                Ok(()) => {
                    let mut seq = self.waiting.pop_front().expect("front existed");
                    seq.state = SeqState::Running;
                    plan.admitted.push(id);
                    self.running.push(seq);
                }
                Err(_) => {
                    // Could not place the prompt. Undo the admit and stop
                    // admitting this step; the request waits.
                    self.cache.free(id);
                    break;
                }
            }
        }

        // Step 2 and 3: try to reserve one decode block per running sequence.
        // If the cache runs out, preempt the least progressed running sequences
        // until the remaining batch fits, then retry.
        loop {
            let needed: usize = self
                .running
                .iter()
                .filter(|s| s.state == SeqState::Running && !s.is_complete())
                .map(|s| self.cache.blocks_needed_for(s.id, 1))
                .sum();

            if needed <= self.cache.free_blocks() || self.running.is_empty() {
                break;
            }

            // Preempt the newest running sequence (the one with the least
            // output, breaking ties by largest id). Recompute-based preemption:
            // we free its blocks entirely and send it back to waiting, where it
            // will re-prefill when readmitted.
            let victim_idx = self
                .running
                .iter()
                .enumerate()
                .filter(|(_, s)| s.state == SeqState::Running)
                .min_by_key(|(_, s)| (s.output.len(), std::cmp::Reverse(s.id)))
                .map(|(i, _)| i);

            match victim_idx {
                Some(i) => {
                    let mut victim = self.running.remove(i);
                    self.cache.free(victim.id);
                    victim.state = SeqState::Preempted;
                    plan.preempted.push(victim.id);
                    self.waiting.push_front(victim);
                }
                None => break, // nothing left to preempt
            }
        }

        // Step 2 (commit): reserve the decode block and record the batch.
        for seq in self.running.iter_mut() {
            if seq.state != SeqState::Running || seq.is_complete() {
                continue;
            }
            match self.cache.append(seq.id, 1) {
                Ok(()) => plan.decode_batch.push(seq.id),
                Err(_) => {
                    // Should not happen after preemption, but stay safe: leave
                    // the sequence in place to be retried next step.
                }
            }
        }

        // Step 4: retire finished sequences and free their blocks.
        let mut still_running = Vec::with_capacity(self.running.len());
        for mut seq in self.running.drain(..) {
            if seq.is_complete() {
                seq.state = SeqState::Finished;
                self.cache.free(seq.id);
                plan.finished.push(seq.id);
            } else {
                still_running.push(seq);
            }
        }
        self.running = still_running;

        plan
    }

    /// Record a generated token for a running sequence. The engine calls this
    /// after the forward pass for every id in `decode_batch`.
    pub fn push_token(&mut self, id: SeqId, token: TokenId, is_eos: bool) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.id == id) {
            seq.output.push(token);
            if is_eos {
                // Force retirement on the next schedule by capping the limit.
                seq.max_new_tokens = seq.output.len();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(blocks: usize, block_size: usize) -> PagedKVCache {
        PagedKVCache::new(blocks, block_size)
    }

    #[test]
    fn admits_up_to_batch_size() {
        let mut sched = Scheduler::new(SchedulerConfig { max_batch_size: 2 }, cache(64, 8));
        for id in 0..5 {
            sched.submit(Sequence::new(id, vec![1, 2, 3], 4));
        }
        let plan = sched.schedule();
        assert_eq!(plan.admitted.len(), 2, "batch cap limits admissions");
        assert_eq!(sched.running_len(), 2);
        assert_eq!(sched.waiting_len(), 3);
    }

    #[test]
    fn batches_decode_across_sequences() {
        let mut sched = Scheduler::new(SchedulerConfig { max_batch_size: 4 }, cache(64, 8));
        for id in 0..3 {
            sched.submit(Sequence::new(id, vec![1, 2], 5));
        }
        let plan = sched.schedule();
        // All three admitted and all three decode together in one batch.
        assert_eq!(plan.admitted.len(), 3);
        assert_eq!(plan.decode_batch.len(), 3);
    }

    #[test]
    fn finished_sequences_retire_and_free_blocks() {
        let mut sched = Scheduler::new(SchedulerConfig::default(), cache(64, 8));
        sched.submit(Sequence::new(1, vec![1, 2], 1));
        sched.schedule();
        let used_before = sched.cache().used_blocks();
        assert!(used_before > 0);

        // Produce the one token it was allowed, then schedule again to retire.
        sched.push_token(1, 9, false);
        let plan = sched.schedule();
        assert_eq!(plan.finished, vec![1]);
        assert_eq!(sched.cache().used_blocks(), 0, "blocks returned on retire");
    }

    #[test]
    fn eos_retires_a_sequence_early() {
        let mut sched = Scheduler::new(SchedulerConfig::default(), cache(64, 8));
        sched.submit(Sequence::new(1, vec![1, 2], 100));
        sched.schedule();
        sched.push_token(1, 0, true); // eos
        let plan = sched.schedule();
        assert_eq!(
            plan.finished,
            vec![1],
            "eos retires before the length limit"
        );
    }

    #[test]
    fn preempts_when_blocks_run_out() {
        // A tiny cache so the decode step cannot grow every sequence. The
        // scheduler must preempt the least progressed one rather than fail.
        // block_size 1 makes every token cost a block, so pressure is exact.
        let mut sched = Scheduler::new(SchedulerConfig { max_batch_size: 8 }, cache(6, 1));
        // Two sequences, prompt length 2 each, uses 4 of 6 blocks on admit.
        sched.submit(Sequence::new(1, vec![1, 2], 50));
        sched.submit(Sequence::new(2, vec![3, 4], 50));
        let plan = sched.schedule();
        assert_eq!(plan.admitted.len(), 2);
        // 4 used, 2 free, both decode (need 2). Fine on step one.
        assert!(plan.preempted.is_empty());

        // Give each a token so they now hold 3 blocks each (6 used, 0 free).
        sched.push_token(1, 5, false);
        sched.push_token(2, 6, false);
        // Next step both need one more block but none are free -> preempt one.
        let plan = sched.schedule();
        assert_eq!(plan.preempted.len(), 1, "exactly one sequence preempted");
        // The preempted one (least output, id 2 by tie-break) returns to wait.
        assert_eq!(plan.preempted[0], 2);
        assert_eq!(sched.waiting_len(), 1);
    }

    #[test]
    fn preempted_sequence_resumes_later() {
        let mut sched = Scheduler::new(SchedulerConfig { max_batch_size: 8 }, cache(6, 1));
        sched.submit(Sequence::new(1, vec![1, 2], 50));
        sched.submit(Sequence::new(2, vec![3, 4], 50));
        sched.schedule();
        sched.push_token(1, 5, false);
        sched.push_token(2, 6, false);
        sched.schedule(); // preempts seq 2

        // Finish seq 1 so its blocks free up, then seq 2 should be readmitted.
        sched.push_token(1, 7, false);
        // Cap seq 1 so it retires.
        sched.running_seq_mut(1).unwrap().max_new_tokens = 2;
        sched.schedule(); // retires seq 1, frees blocks
        let plan = sched.schedule(); // readmits seq 2
        assert!(
            plan.admitted.contains(&2),
            "preempted sequence must be readmitted once blocks free, plan {plan:?}"
        );
    }

    #[test]
    fn admission_blocks_when_prompt_does_not_fit() {
        // A prompt larger than the whole cache can never be admitted, and must
        // not crash the scheduler or starve other work indefinitely.
        let mut sched = Scheduler::new(SchedulerConfig::default(), cache(2, 1));
        sched.submit(Sequence::new(1, vec![1, 2, 3, 4, 5], 1)); // needs 5 blocks
        let plan = sched.schedule();
        assert!(plan.admitted.is_empty(), "oversized prompt is not admitted");
        assert_eq!(sched.running_len(), 0);
        assert_eq!(
            sched.cache().used_blocks(),
            0,
            "no blocks leaked on failure"
        );
    }

    #[test]
    fn idle_when_no_work() {
        let mut sched = Scheduler::new(SchedulerConfig::default(), cache(8, 4));
        assert!(sched.schedule().is_idle());
    }
}
