# Architecture

forge-infer is organised so each serving technique lives in its own module with its own tests, and the model is a swappable trait. This page is the map: the modules, the request lifecycle, the one design decision that makes the whole policy testable, and the concurrency model.

## Module map

| Module | File | Key symbols | Responsibility |
| --- | --- | --- | --- |
| `model` | `src/model.rs` | `Model`, `TinyTransformer`, `StepLogits` | The model trait and the deterministic stand-in. |
| `tokeniser` | `src/tokeniser.rs` | `encode`, `decode`, `decode_one`, `EOS_TOKEN` | Reversible byte-level codec. |
| `paged_cache` | `src/paged_cache.rs` | `PagedKVCache`, `BlockTable`, `CacheError` | Block-based KV allocation and block tables. |
| `scheduler` | `src/scheduler.rs` | `Scheduler::schedule`, `StepPlan`, `Sequence`, `SeqState` | Continuous batching, admission, preemption. |
| `speculative` | `src/speculative.rs` | `SpeculativeDecoder::step`, `SpeculationResult` | Draft-then-verify decoding with an acceptance test. |
| `engine` | `src/engine.rs` | `Engine::step`, `Engine::run_to_completion` | The loop joining the scheduler to the model. |
| `server` | `src/server.rs` | `router`, `AppState`, `completions` | The axum HTTP server and OpenAI-compatible endpoint. |

## The request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant H as axum handler
    participant T as Tokeniser
    participant Sc as Scheduler
    participant K as PagedKVCache
    participant M as Model

    C->>H: POST /v1/completions {prompt, max_tokens, stream}
    H->>T: encode(prompt)
    T-->>H: token ids
    H->>Sc: submit(Sequence)
    loop one iteration per decode step
        Sc->>Sc: schedule() builds a StepPlan
        Sc->>K: admit + reserve blocks
        alt blocks exhausted
            Sc->>K: free blocks of least-progressed sequence
            Sc->>Sc: preempt and requeue
        end
        Sc->>M: forward(context) for each id in decode_batch
        M-->>Sc: logits, then argmax token
        Sc->>Sc: push_token, retire on eos or limit
    end
    H->>T: decode(tokens)
    H-->>C: SSE stream of token deltas, then [DONE]
```

## The decision that makes it testable

The scheduler and the engine are deliberately split. `Scheduler::schedule` (`src/scheduler.rs`) mutates only the cache and the waiting and running sets, and returns its decisions as a plain `StepPlan` value. `Engine::step` (`src/engine.rs`) is the only thing that ever calls `Model::forward`. Two payoffs:

1. **The scheduling policy is unit-testable without a model or a GPU.** The tests in `src/scheduler.rs` construct a scheduler, submit sequences, call `schedule()`, and assert directly on the returned `StepPlan` and on cache block counts. `preempts_when_blocks_run_out` and `admission_blocks_when_prompt_does_not_fit` run with no forward pass at all.
2. **The model is swappable.** Because the engine talks to a `Model` trait, replacing `TinyTransformer` with a real backend is a matter of implementing one method, `forward(&self, context: &[TokenId]) -> StepLogits`.

The cost of the split is one extra indirection in the engine loop: `schedule()` returns the batch, then `step()` walks `plan.decode_batch` calling the model. I pay that every time for a policy I can test in isolation. The tempting alternative, a single `tick()` that schedules and runs the forward pass together, would tangle the two and force every scheduler test to stand up a model.

## The Model trait

```rust
pub trait Model: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn eos_token(&self) -> TokenId;
    fn forward(&self, context: &[TokenId]) -> StepLogits;
}
```

`StepLogits` carries the next-token logits and offers `argmax` for greedy decoding and `prob_of(token)` for the speculative acceptance test. `TinyTransformer::forward` maps a trailing eight-token window through a splitmix-style hash (`hash_context`) to a peaked logits distribution. It is a pure function of the context, which is exactly what lets the cache, the scheduler and the decoder be verified deterministically. The draft variant from `TinyTransformer::draft` shifts its peak on a deterministic quarter of contexts, so a draft and a target agree on most tokens and disagree on some, the realistic regime for speculation.

## Concurrency model

The server shares one `Arc<dyn Model>` across requests so the model weights stay resident, and gives each request its own `Engine` with its own `PagedKVCache` and scheduler (`AppState::new_engine` in `src/server.rs`). That keeps the example readable, at the cost of not yet sharing one long-lived engine loop across live HTTP traffic.

A production deployment would route all requests into a single engine loop so the continuous batching scheduler interleaves decode steps across concurrent requests. That loop is exactly what `Engine::step` implements, and the benchmark's continuous-batching path drives 64 requests through one engine to demonstrate it. Closing that gap on the server side is the first item on the [Roadmap](Roadmap-and-Limitations).

## Failure modes to know

- An oversized prompt (more blocks than the cache holds) is never admitted. `schedule()` rolls the admit back cleanly and the request waits; `admission_blocks_when_prompt_does_not_fit` pins this down.
- Under block pressure the scheduler preempts. A preempted request is invisible to the client: it still completes, it just yields the engine for a while.
- `run_to_completion` carries a one-million-iteration guard so a pathological loop cannot hang the process.

## Going deeper

Each module has its own page: [Engine](Engine), [Model-and-Tokeniser](Model-and-Tokeniser), [HTTP-Server](HTTP-Server), [Paged-KV-Cache](Paged-KV-Cache), [Continuous-Batching](Continuous-Batching) and [Speculative-Decoding](Speculative-Decoding). The choices behind the split are on [Design-Decisions](Design-Decisions), and the full symbol list is on [API-Reference](API-Reference).

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
