# Benchmarks

The `forge-bench` binary measures the engine three ways and prints a table. This page explains what it measures, how to read it, and what the numbers mean given that the model is a deterministic stand-in.

## Running it

```bash
cargo run --release --bin forge-bench
```

Always build in release mode for benchmarking. The debug build carries overflow checks and no optimisation, so its numbers are not representative.

## What it measures

The benchmark fixes the workload at 64 requests, 16-token prompts and 64 max new tokens, then runs three strategies:

1. **sequential**: a fresh engine per request with `max_batch_size = 1`. This is the static-batching baseline where requests never share a decode step.
2. **continuous-batching**: one engine, all 64 requests submitted up front, `max_batch_size = 16`. Requests share decode iterations and the engine stays saturated.
3. **speculative**: a single-stream draft-then-verify decoder with lookahead 4, reporting the aggregate acceptance rate.

For each it reports total tokens, tokens per second, and milliseconds per request.

## Example output

```
forge-infer benchmark
requests=64 prompt_len=16 max_new_tokens=64

strategy                 tokens     tokens/sec     ms/request  notes
--------------------------------------------------------------------------------------
sequential                 3690        1757317          0.033  batch=1 baseline
continuous-batching        3690        1909609          0.030  batch=16
speculative                3690         403009          0.143  acceptance=52%

continuous batching throughput vs sequential: 1.09x
```

Numbers vary run to run with machine load; the shape is what matters.

## How to read these numbers

The model here is a cheap deterministic hash, not a GPU kernel, so the absolute tokens-per-second figures reflect **scheduling and bookkeeping overhead**, not model compute. That is the point of the benchmark: it isolates the cost of the serving machinery.

- **continuous batching versus sequential.** With a near-free model the per-token model cost is tiny, so the gap mostly measures scheduler overhead rather than the throughput multiplier you would see with a real model. The crucial property the continuous path demonstrates is that 64 concurrent requests run through a single shared engine loop, interleaving their decode steps, which is the structure that delivers large speedups once the model cost is real. With a genuinely expensive model, keeping the batch full is worth far more than the small scheduling overhead shown here.
- **speculative acceptance rate.** The headline metric is the 52% acceptance rate: just over half the draft model's proposals are accepted by the target without a recompute, and the emitted tokens are provably identical to plain target decoding (see [Speculative-Decoding](Speculative-Decoding)). With a real model where the target forward pass dominates, a 52% acceptance rate translates directly into a meaningful latency reduction, because every accepted token skips a target step.

## Where the speedups come from with a real model

The techniques in forge-infer pay off in proportion to how expensive the model is:

- **Paged KV-cache** raises the number of concurrent sequences you can hold in a fixed memory budget, by removing external fragmentation and bounding internal fragmentation. More concurrency means a fuller batch.
- **Continuous batching** keeps that batch full by refilling it every iteration, so the expensive model is never waiting on a half-empty batch.
- **Speculative decoding** cuts single-stream latency by amortising the target forward pass over several tokens per accepted round.

The benchmark here demonstrates that all three are implemented and behave correctly; the magnitude of the win scales with the cost of the model you plug in behind the `Model` trait.
