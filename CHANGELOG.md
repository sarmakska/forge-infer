# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `PagedKVCache::peak_blocks`: a lifetime high-water mark of simultaneously used blocks, the true minimum cache size a workload needed. The benchmark surfaces it as `peak_kv_blocks` (75 on the standard 64-request workload, against a 320-block naive worst case), making the memory argument for paging a single measured number.
- `PagedKVCache`: a block-based KV-cache allocator with per-sequence block tables, lazy block growth, allocate and free, an out-of-blocks signal and a largest-sequence eviction policy. Internal fragmentation is bounded by the block size and external fragmentation is eliminated.

### Changed

- `Engine`: renamed the awkward `scheduler_mut_run(prompt_len, max_new)` to `run_one()`. The previous name was misleading and both arguments were ignored, since the engine already holds the prompt and limit from `submit`.
- Removed the dead `Scheduler::take_finished` stub, which always returned an empty vector. Callers read `StepPlan::finished` instead.
- `ContinuousBatchingScheduler`: iteration-level scheduling with admission control, decode batching across sequences, and recompute-based preemption with resume when blocks run out.
- `SpeculativeDecoder`: a draft-then-verify loop using the standard rejection-sampling acceptance test, with an exact-output guarantee against plain target decoding and a correct single-token fallback on rejection.
- `Engine`: the loop that joins the scheduler to the model, running batched decode steps and feeding tokens back.
- `Model` trait and `TinyTransformer`, a small fixed-weight deterministic model that keeps the build fast and the tests reproducible.
- Reversible byte-level tokeniser.
- axum HTTP server with `POST /generate`, an OpenAI-compatible `POST /v1/completions` with Server-Sent Events streaming, and `GET /healthz`.
- `forge-bench` binary measuring tokens per second and latency for sequential, continuous-batching and speculative strategies.
- Test suite covering paged cache allocate, free, fragmentation and eviction, scheduler batching and preemption decisions, speculative acceptance correctness, and an integration test that boots the server and streams a completion.
- Documentation: product README with a Mermaid architecture diagram, and a wiki covering Architecture, Paged-KV-Cache, Continuous-Batching, Speculative-Decoding, Benchmarks and Troubleshooting.
- CI workflow running format check, clippy, build and test on push and pull request.
