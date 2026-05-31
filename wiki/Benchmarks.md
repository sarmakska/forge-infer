# Benchmarks

The `forge-bench` binary (`src/bin/bench.rs`) measures the engine three ways and prints a table. This page explains what it measures, gives the real numbers from this machine, and is honest about what they do and do not mean given that the model is a deterministic stand-in.

## Running it

```bash
cargo run --release --bin forge-bench
```

Always build in release mode for benchmarking. The debug build carries overflow checks and no optimisation, so its numbers are not representative.

## What it measures

The benchmark fixes the workload at 64 requests, 16-token prompts and 64 max new tokens, then runs three strategies:

1. **sequential**: a fresh engine per request with `max_batch_size = 1`. The static-batching baseline where requests never share a decode step.
2. **continuous-batching**: one engine, all 64 requests submitted up front, `max_batch_size = 16`. Requests share decode iterations and the engine stays saturated.
3. **speculative**: a single-stream draft-then-verify decoder with lookahead 4, reporting the aggregate acceptance rate.

For each it reports total tokens, tokens per second, and milliseconds per request.

## Real numbers

Measured on an **Apple M3 Pro** (macOS 25.3, `rustc 1.96.0`) with `cargo run --release --bin forge-bench`. Three consecutive runs, to show the spread:

```
strategy                 tokens     tokens/sec     ms/request  notes
--------------------------------------------------------------------------------------
sequential                 3690        1830357          0.032  batch=1 baseline
continuous-batching        3690        2061789          0.028  batch=16
speculative                3690         567856          0.102  acceptance=52%
continuous batching throughput vs sequential: 1.13x

sequential                 3690        1881333          0.031  batch=1 baseline
continuous-batching        3690        2140164          0.027  batch=16
speculative                3690         588642          0.098  acceptance=52%
continuous batching throughput vs sequential: 1.14x

sequential                 3690        1924883          0.030  batch=1 baseline
continuous-batching        3690        2070513          0.028  batch=16
speculative                3690         609066          0.095  acceptance=52%
continuous batching throughput vs sequential: 1.08x
```

Representative figures: sequential around 1.88M tokens/sec, continuous batching around 2.07M tokens/sec, speculative around 0.59M tokens/sec with a 52% acceptance rate. The continuous-over-sequential ratio sits around 1.1x on this near-free model. Numbers drift a few percent run to run with machine load; the shape is what carries.

## How to read these numbers

The model here is a cheap deterministic hash, not a GPU kernel, so the absolute tokens-per-second figures reflect **scheduling and bookkeeping overhead**, not model compute. That is the point of the benchmark: it isolates the cost of the serving machinery.

- **Continuous batching versus sequential.** With a near-free model the per-token model cost is tiny, so the gap mostly measures scheduler overhead rather than the multiplier you would see with a real model. The property the continuous path demonstrates is that 64 concurrent requests run through a single shared engine loop, interleaving their decode steps. With a genuinely expensive model, keeping the batch full is worth far more than the small scheduling overhead shown here, and that ratio grows accordingly.
- **Speculative acceptance rate.** The headline metric is the 52% acceptance rate: just over half the draft's proposals are accepted by the target without a recompute, and the emitted tokens are provably identical to plain target decoding (see [Speculative-Decoding](Speculative-Decoding)). With a real model where the target forward pass dominates, a 52% acceptance rate translates directly into a latency reduction, because every accepted token skips a target step. Here, where both models are near-free, speculative is *slower* in wall-clock terms because it runs more forward passes per emitted token; that is expected and is exactly why the acceptance rate, not the tokens/sec, is the figure to carry across to a real model.

## Where the speedups come from with a real model

The techniques pay off in proportion to how expensive the model is:

- **Paged KV-cache** raises the number of concurrent sequences you can hold in a fixed memory budget, by removing external fragmentation and bounding internal fragmentation. More concurrency means a fuller batch.
- **Continuous batching** keeps that batch full by refilling it every iteration, so the expensive model is never waiting on a half-empty batch.
- **Speculative decoding** cuts single-stream latency by amortising the target forward pass over several tokens per accepted round.

The benchmark here demonstrates that all three are implemented and behave correctly; the magnitude of the win scales with the cost of the model you plug in behind the `Model` trait.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
