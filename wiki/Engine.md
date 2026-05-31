# Engine

The engine is the loop that joins the scheduler to the model. It is the smallest module in the project (`src/engine.rs`, about 170 lines including tests) and the most important one to understand, because it is where a scheduling decision turns into a forward pass and a token comes back. This page walks the loop line by line, explains the two entry points, and lays out the failure modes the loop has to survive.

## What the engine owns

```rust
pub struct Engine {
    scheduler: Scheduler,
    model: Arc<dyn Model>,
}
```

One `Scheduler` (which itself owns the `PagedKVCache`) and one `Arc<dyn Model>`. The `Arc` is deliberate: the HTTP server shares a single model across every request so the weights stay resident, while each request gets its own engine and therefore its own cache and queues. The engine is single-threaded; concurrency is the server's job, not the engine's.

`Engine::new` takes the model, a `SchedulerConfig` and a `PagedKVCache`, and wires them together. That is the only constructor; there is no builder, because there are exactly three things to supply.

## The one iteration: `Engine::step`

`step` is the heart of the project. It does three things in order: schedule, forward, feed back.

```rust
pub fn step(&mut self) -> Vec<(u64, TokenId, bool)> {
    let plan = self.scheduler.schedule();
    let mut emitted = Vec::with_capacity(plan.decode_batch.len());

    for id in &plan.decode_batch {
        let context = {
            let seq = self.scheduler.running_seq_mut(*id)
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
```

The shape of this function carries the central design decision of the whole project. The scheduler runs first and returns a `StepPlan`. The engine then iterates `plan.decode_batch`, the ids the scheduler has decided will decode this step, and for each one it rebuilds the context (prompt plus output so far), calls `Model::forward` exactly once, takes the greedy `argmax`, and pushes the token back into the scheduler. The scheduler never sees the model; the engine never makes a scheduling decision. That separation is what makes the scheduler unit-testable without a GPU, and it is argued in full on the [Architecture](Architecture) and [Design-Decisions](Design-Decisions) pages.

```mermaid
flowchart LR
    SCH["scheduler.schedule()<br/>-> StepPlan"] --> ITER["for id in decode_batch"]
    ITER --> CTX["rebuild context<br/>prompt + output"]
    CTX --> FWD["model.forward(context)"]
    FWD --> AM["logits.argmax()"]
    AM --> PB["scheduler.push_token(id, tok, eos)"]
    PB --> ITER
    PB --> EM["emit (id, token, eos)"]

    classDef sky fill:#0d1117,stroke:#38bdf8,color:#f5f7fa;
    classDef cyan fill:#0d1117,stroke:#22d3ee,color:#f5f7fa;
    classDef em fill:#0d1117,stroke:#34d399,color:#f5f7fa;
    class SCH,ITER sky;
    class CTX,FWD,AM cyan;
    class PB,EM em;
```

### Why the context is rebuilt every step

The engine clones `prompt` and appends `output` on every decode step, then hands the whole slice to `forward`. That is O(context length) work per token, which a production engine would never do: a real backend keeps the key and value tensors for past tokens in the KV-cache and attends over them, computing only the new token's contribution. forge-infer recomputes from scratch because the `TinyTransformer` is a pure hash of a trailing window and has no persistent state to carry. The block table that a real backend would gather over is already built and maintained by the cache (`PagedKVCache::block_table`); wiring a stateful backend to it is the extension described on [Writing-a-Model-Backend](Writing-a-Model-Backend). This recompute is listed honestly as a limitation on the [Roadmap-and-Limitations](Roadmap-and-Limitations) page.

## Running to completion: `run_to_completion`

The benchmark and the integration tests do not stream; they want every submitted request finished and collected. `run_to_completion` drives `step` until the scheduler reports no work left:

```rust
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
        if guard > max_iterations { break; }
    }
    // collect into Vec<GenerationOutput>, sorted by id
}
```

Two details matter. The output is accumulated per id in a `HashMap` and then sorted by id before returning, so `run_to_completion` is deterministic regardless of the order the scheduler happens to retire sequences. And the loop carries a one-million-iteration guard. That guard is a backstop against a pathological model whose `eos_token` never appears and whose sequences never hit their limit; it cannot deadlock the process. In normal operation it is never reached, because every sequence has a `max_new_tokens` cap and the scheduler retires it on the step after it hits the cap.

## The convenience wrapper the server uses

```rust
pub fn scheduler_mut_run(&mut self, _prompt_len: usize, _max_new: usize) -> Vec<TokenId> {
    let outputs = self.run_to_completion();
    outputs.into_iter().next().map(|o| o.tokens).unwrap_or_default()
}
```

The HTTP handlers submit exactly one prompt and want one token stream back. `scheduler_mut_run` runs to completion and returns the first (and only) output's tokens, or an empty vector if nothing was produced. The `_prompt_len` and `_max_new` parameters are accepted for call-site readability and are intentionally unused; the engine already knows the limits from `submit`. They are prefixed with an underscore so clippy stays quiet.

## Submitting work

```rust
pub fn submit(&mut self, id: u64, prompt: Vec<TokenId>, max_new_tokens: usize) {
    self.scheduler.submit(Sequence::new(id, prompt, max_new_tokens));
}
```

`submit` is a thin forward to the scheduler. The caller owns the id space; the server uses `1` for its single per-request sequence, the benchmark uses `0..NUM_REQUESTS`. Ids only need to be unique within one engine's lifetime.

## Failure modes

- **A decode-batch id that is not running.** `step` calls `running_seq_mut(*id).expect(...)`. This `expect` encodes an invariant: the scheduler only ever puts running sequence ids into `decode_batch`. If it ever fired it would mean a scheduler bug, not a runtime condition, which is why it panics rather than returns an error.
- **A model that never emits eos.** Handled by the iteration guard in `run_to_completion`. The sequence still terminates because of its `max_new_tokens` limit; the guard only matters if a caller sets an unbounded limit.
- **An empty decode batch.** If the scheduler returns an idle plan (no work, or everything preempted), `step` does no forward passes and returns an empty vector. `run_to_completion` keeps looping only while `has_work()` is true, so an engine with nothing submitted returns immediately.

## What the tests prove

The tests in `src/engine.rs` exercise the loop end to end against the `TinyTransformer`:

- `single_request_produces_requested_tokens`: one prompt produces at most its limit and at least one token.
- `many_requests_all_complete_under_pressure`: ten requests through a 32-block cache and a batch size of 2 forces interleaving, preemption and resume, and every request still completes. This is the test that proves the engine and scheduler cooperate correctly under memory pressure.
- `output_is_deterministic`: two engines given the same prompt produce byte-identical output, which is the property the whole design rests on.

## See also

- [Architecture](Architecture) for the module map and the scheduler/engine split.
- [Continuous-Batching](Continuous-Batching) for what `schedule()` decides before the engine runs the forward pass.
- [Writing-a-Model-Backend](Writing-a-Model-Backend) for replacing `TinyTransformer` with real weights.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
