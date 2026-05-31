# forge-infer

A minimal LLM inference server that implements the real serving techniques: paged KV-cache, continuous batching and speculative decoding.

I built forge-infer to be the thing I wish I had when I first tried to understand how a modern inference server works. The papers describe paging and continuous batching at a high level, the production engines bury them under thousands of lines of CUDA, and the toy examples skip the hard parts entirely. forge-infer sits in between: the serving algorithms are implemented for real and tested, and the model behind them is a small deterministic stand-in so the whole stack builds in seconds and behaves the same way every run.

## What you will find here

- **[Architecture](Architecture)**: the module map, the request lifecycle, and the design decision that makes the whole thing testable without a GPU.
- **[Paged-KV-Cache](Paged-KV-Cache)**: how block-based KV allocation works, why it removes fragmentation, and how the allocator is implemented.
- **[Continuous-Batching](Continuous-Batching)**: iteration-level scheduling, admission control and preemption, with the exact decision rules.
- **[Speculative-Decoding](Speculative-Decoding)**: the draft-then-verify loop, the rejection-sampling acceptance test, and the proof-by-test that the output is exact.
- **[Benchmarks](Benchmarks)**: what forge-bench measures, how to read it, and the numbers on a typical machine.
- **[Troubleshooting](Troubleshooting)**: build issues, runtime questions and the answers.

## The thirty-second tour

```bash
cargo test                                   # 37 tests across the serving stack
cargo run --release --bin forge-infer        # start the server on 127.0.0.1:8080
cargo run --release --bin forge-bench        # print the throughput table
```

```bash
curl -s localhost:8080/generate -d '{"prompt":"hello forge","max_tokens":24}'
curl -sN localhost:8080/v1/completions -d '{"prompt":"stream me","max_tokens":20,"stream":true}'
```

## Design philosophy

Every component a real engine has is present and named the same way it is in the literature: a `Model` trait, a `PagedKVCache` with a block table, a `ContinuousBatchingScheduler` that produces a `StepPlan` each iteration, and a `SpeculativeDecoder` with an acceptance test. The one simplification is the model: it is a deterministic hash-based `TinyTransformer` rather than a stack of attention layers. That choice is deliberate. It keeps the build fast, it makes the speculative acceptance test assertable, and it means the project demonstrates the serving systems rather than re-implementing PyTorch. To serve a real model you would implement the `Model` trait against your weights; nothing else in the stack would change.
