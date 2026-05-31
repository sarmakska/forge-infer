# Roadmap and Limitations

forge-infer is a teaching engine for three serving algorithms. This page is honest about what it is not, and opinionated about what I will and will not add. Real projects say what they are not.

## What this is not

- **Not a production inference server.** There are no real weights, no attention maths and no GPU kernels. The model is a deterministic hash that returns a reproducible byte stream, not language. If you want prose out, implement `Model::forward` against your own weights.
- **Not a model library.** The interesting code is the cache, the scheduler and the decoder. The model exists only to give them something to call.
- **Not a benchmark of model throughput.** Because the model is near-free, `forge-bench` measures the cost of the serving machinery, not the cost of inference. Read its numbers for shape, not magnitude (see [Benchmarks](Benchmarks)).

## Current limitations

- The HTTP layer spins up a fresh `Engine` per request over a shared `Arc<dyn Model>`. That keeps the example readable but means concurrent HTTP requests do not yet share one long-lived engine loop on the server side. The continuous-batching path is exercised by the benchmark, which drives 64 requests through a single engine.
- KV state is recomputed from the full context on each `forward` rather than carried across calls. A real backend would attend over the physical blocks `PagedKVCache::block_table` hands it.
- Decoding is greedy argmax only. No temperature, top-p or top-k sampling on the server path.
- Preemption is recompute-based; there is no KV swap to host memory. That is a deliberate choice, explained below.

## Roadmap

### Will add

- **A single shared server-side engine loop**, so live HTTP traffic gets real continuous batching rather than per-request engines. This is the highest-value gap and the most natural next step, since `Engine::step` already implements the loop.
- **Prefix sharing with copy-on-write block tables.** A block table is exactly the structure that makes shared prompt prefixes cheap: two sequences point at the same physical blocks until one of them diverges. This is the next obvious use of the allocator already in place.
- **A `candle` feature flag behind the `Model` trait** for anyone who wants real text out, kept optional so the default build stays fast and dependency-light.

### Will not add

- **Distributed or multi-GPU serving.** Out of scope for a single-author teaching engine.
- **A model zoo or downloadable weights.** The model is a placeholder by design.
- **A web UI.** The repository is about reading three algorithms end to end. A UI would be noise.

The rule for the roadmap is simple: anything that makes the three serving algorithms clearer or more correct is fair game; anything that buries them under surface area is not.

## On recompute versus KV swapping

A note on the one preemption trade-off, since it comes up. vLLM can either recompute a preempted sequence or swap its KV blocks out to CPU memory and back. Swapping saves the recompute but adds a copy path, a host memory pool and a second eviction policy. For a teaching engine the recompute path is the one worth showing: it is a dozen lines, it provably never deadlocks because freeing a sequence releases at least the blocks the step needs, and it makes the cost of preemption legible. I will not add swapping; the added surface area would obscure the very thing the scheduler exists to teach.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
