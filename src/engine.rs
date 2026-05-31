//! The inference engine: the loop that drives the scheduler and the model.
//!
//! The engine owns the scheduler (and through it the KV-cache) and a model. It
//! exposes a synchronous `run_to_completion` used by the benchmark and a
//! token-streaming API used by the HTTP server. The engine is where scheduling
//! decisions meet forward passes: every iteration it asks the scheduler for a
//! [`StepPlan`], runs one forward pass for each sequence in the decode batch,
//! and feeds the generated tokens back.

use crate::model::{Model, TokenId};
use crate::paged_cache::PagedKVCache;
use crate::scheduler::{Scheduler, SchedulerConfig, Sequence};
use std::collections::HashMap;
use std::sync::Arc;

/// The result of running one request to completion.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub id: u64,
    pub tokens: Vec<TokenId>,
}

/// A single-threaded engine driving one scheduler and one model. The HTTP layer
/// wraps this behind a tokio task and a channel.
pub struct Engine {
    scheduler: Scheduler,
    model: Arc<dyn Model>,
}

impl Engine {
    pub fn new(model: Arc<dyn Model>, config: SchedulerConfig, cache: PagedKVCache) -> Self {
        Self {
            scheduler: Scheduler::new(config, cache),
            model,
        }
    }

    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }

    /// Submit a prompt and get back an id to track it.
    pub fn submit(&mut self, id: u64, prompt: Vec<TokenId>, max_new_tokens: usize) {
        self.scheduler
            .submit(Sequence::new(id, prompt, max_new_tokens));
    }

    /// Run a single engine iteration: schedule, forward, collect tokens.
    /// Returns the `(id, token)` pairs produced this step. The caller streams
    /// or accumulates them.
    pub fn step(&mut self) -> Vec<(u64, TokenId, bool)> {
        let plan = self.scheduler.schedule();
        let mut emitted = Vec::with_capacity(plan.decode_batch.len());

        for id in &plan.decode_batch {
            // Build the context for this sequence: prompt + output so far.
            let context = {
                let seq = self
                    .scheduler
                    .running_seq_mut(*id)
                    .expect("decode_batch ids are running");
                let mut ctx = seq.prompt.clone();
                ctx.extend_from_slice(&seq.output);
                ctx
            };
            let logits = self.model.forward(&context);
            let token = logits.argmax();
            let is_eos = token == self.model.eos_token();
            self.scheduler.push_token(*id, token, is_eos);
            emitted.push((*id, token, is_eos));
        }
        emitted
    }

    /// Run the currently submitted work to completion and return the output
    /// tokens of the single request whose prompt length and limit are given.
    /// This is the convenience the HTTP layer uses: it submits one prompt then
    /// drains it. `_prompt_len` and `_max_new` are accepted for call-site
    /// clarity; the engine already knows the limits from `submit`.
    pub fn scheduler_mut_run(&mut self, _prompt_len: usize, _max_new: usize) -> Vec<TokenId> {
        let outputs = self.run_to_completion();
        outputs
            .into_iter()
            .next()
            .map(|o| o.tokens)
            .unwrap_or_default()
    }

    /// Drive the engine until every submitted request finishes, collecting each
    /// request's full output. Used by the benchmark and integration tests.
    pub fn run_to_completion(&mut self) -> Vec<GenerationOutput> {
        let mut outputs: HashMap<u64, Vec<TokenId>> = HashMap::new();
        let mut guard = 0usize;
        let max_iterations = 1_000_000;

        while self.scheduler.has_work() {
            let emitted = self.step();
            for (id, token, _eos) in emitted {
                outputs.entry(id).or_default().push(token);
            }
            guard += 1;
            if guard > max_iterations {
                break;
            }
        }

        let mut result: Vec<GenerationOutput> = outputs
            .into_iter()
            .map(|(id, tokens)| GenerationOutput { id, tokens })
            .collect();
        result.sort_by_key(|o| o.id);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TinyTransformer;

    fn engine(blocks: usize, batch: usize) -> Engine {
        let model: Arc<dyn Model> = Arc::new(TinyTransformer::new(64, 4));
        Engine::new(
            model,
            SchedulerConfig {
                max_batch_size: batch,
            },
            PagedKVCache::new(blocks, 8),
        )
    }

    #[test]
    fn single_request_produces_requested_tokens() {
        let mut eng = engine(64, 4);
        eng.submit(1, vec![1, 2, 3], 5);
        let out = eng.run_to_completion();
        assert_eq!(out.len(), 1);
        assert!(out[0].tokens.len() <= 5);
        assert!(!out[0].tokens.is_empty());
    }

    #[test]
    fn many_requests_all_complete_under_pressure() {
        // More requests than batch slots and a small cache forces the scheduler
        // to interleave, preempt and resume. Every request must still finish.
        let mut eng = engine(32, 2);
        for id in 0..10 {
            eng.submit(id, vec![1, 2, 3], 6);
        }
        let out = eng.run_to_completion();
        assert_eq!(out.len(), 10, "every request must complete");
        for o in &out {
            assert!(!o.tokens.is_empty());
        }
    }

    #[test]
    fn output_is_deterministic() {
        let mut a = engine(64, 4);
        let mut b = engine(64, 4);
        a.submit(1, vec![5, 6, 7], 8);
        b.submit(1, vec![5, 6, 7], 8);
        assert_eq!(
            a.run_to_completion()[0].tokens,
            b.run_to_completion()[0].tokens
        );
    }
}
