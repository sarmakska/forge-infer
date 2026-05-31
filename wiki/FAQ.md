# FAQ

Short answers to the questions people actually ask about forge-infer. Each one links on to the page with the full story.

## Is this a real inference server I can put in production?

No, and it does not pretend to be. There are no real weights, no attention maths and no GPU kernels. The model is a deterministic hash that returns a reproducible byte stream. The serving stack around it (cache, scheduler, decoder, engine, server) is real and is the deliverable; the model is a placeholder you swap out by implementing one trait method. See [Roadmap-and-Limitations](Roadmap-and-Limitations) and [Security-Model](Security-Model).

## Why is the generated text gibberish?

Because the model is a hash, not a language model. The output is a reproducible byte stream that round-trips through the tokeniser as valid UTF-8, but it is not prose. This is by design: a deterministic model is what makes the speculative acceptance test assertable and the benchmarks reproducible. To get real text, implement `Model::forward` against real weights ([Writing-a-Model-Backend](Writing-a-Model-Backend)).

## Then what is the point if it does not generate language?

The point is the three serving systems that decide how fast a real engine runs: the paged KV-cache, the continuous-batching scheduler, and the speculative decoder. These are implemented for real, with the awkward cases handled (out-of-blocks preemption and resume, the rejection-sampling acceptance test). Most tutorials mock exactly these parts and show you a toy model instead. forge-infer inverts that: the model is the toy, the systems are real.

## Why not use candle or tch for a real tiny model?

Two reasons, argued in full on [Design-Decisions](Design-Decisions). First, the speculative acceptance test needs `p(t)/q(t)` to be bit-for-bit reproducible across two model instances, and floating-point attention is not stable enough to assert equality on. Second, a real model adds a multi-second compile and a heavy dependency tree between a reader and the code. The hash is pure, exact and fast.

## How do I plug in my own model?

Implement the four-method `Model` trait, the only one that does work being `forward(&self, context: &[TokenId]) -> StepLogits`. Build an `AppState` with your model in place of `TinyTransformer`. Nothing in the cache, scheduler, decoder or engine changes. Full walkthrough on [Writing-a-Model-Backend](Writing-a-Model-Backend).

## Does the OpenAI client really work against this?

Yes. Point the client's `base_url` at `http://localhost:8080/v1` and call completions, including `stream: true`. The endpoint matches the OpenAI completion shape and emits SSE deltas ending in `[DONE]`. There is no auth, so any non-empty API key works. Recipe on [Examples-and-Recipes](Examples-and-Recipes).

## Why does streaming arrive all at once?

The model is near-instant, so the server generates the whole completion up front and then emits one SSE event per token. The events arrive together because there is no per-token compute to space them out. With a real model the per-token cost would stagger them. The wire format is genuine SSE; use `curl -N` to see the individual `data:` lines. See [HTTP-Server](HTTP-Server).

## What happens when the cache runs out of blocks?

The scheduler preempts. It frees the least-progressed running sequence's blocks entirely and sends it back to the waiting queue, where it re-prefills when readmitted. This is recompute-based preemption; it never deadlocks because freeing a sequence always releases at least the blocks the step needs. It is invisible to the client: the request still completes, it just yields the engine for a while. See [Continuous-Batching](Continuous-Batching).

## Why preempt the newest sequence and not the oldest?

Preempting the least-progressed sequence minimises wasted recompute: a sequence with two tokens of output loses two tokens when it resumes, one with two hundred would lose two hundred. It is the shortest-remaining-time intuition. Detail on [Continuous-Batching](Continuous-Batching).

## Why recompute instead of swapping KV to host memory?

Swapping saves the recompute but adds a copy path, a host memory pool and a second eviction policy, roughly doubling the cache's surface area, to demonstrate a latency optimisation that only matters once the model is genuinely expensive. For a teaching engine the recompute path is a dozen legible lines that provably never deadlock. Full argument on [Design-Decisions](Design-Decisions).

## Is speculative decoding an approximation?

No. It is exact. The rejection-sampling acceptance test guarantees the emitted tokens are distributed identically to plain sampling from the target model. The test `speculative_output_matches_plain_target_decoding` checks this token for token with identical draft and target. It is a reorganisation of the same computation, not a shortcut that trades quality for speed. See [Speculative-Decoding](Speculative-Decoding).

## Why is speculative decoding slower in the benchmark?

Because the model is near-free. Speculative decoding runs more forward passes per emitted token (the draft proposes, the target verifies), and when forward passes cost almost nothing, doing more of them is slower in wall-clock terms. The figure that carries to a real model is the 52% acceptance rate, not the tokens per second: with an expensive target, every accepted token skips a target step. See [Benchmarks](Benchmarks).

## What do the benchmark numbers actually mean?

They isolate the cost of the serving machinery, not model compute, because the model is near-free. Read them for shape, not magnitude. The continuous-vs-sequential ratio (around 1.1x here) is mostly scheduler overhead; with a real model, keeping the batch full dwarfs the bookkeeping and that ratio grows. The acceptance rate is the transferable number. See [Benchmarks](Benchmarks).

## How many tests are there, and do they need a GPU?

37 tests (33 unit, 4 integration), all running in well under a second on CPU with no GPU. The deterministic model is what lets the cache, scheduler and decoder be tested as exact facts. See [Testing-Strategy](Testing-Strategy).

## What Rust version do I need?

Rust 1.96 or newer, set as `rust-version` in `Cargo.toml`. The code uses standard-library methods like `div_ceil` and `is_multiple_of` that older toolchains lack. See [Troubleshooting](Troubleshooting).

## Can I change the port?

Set `FORGE_ADDR`, for example `FORGE_ADDR=127.0.0.1:9000 cargo run --bin forge-infer`. Read in `src/main.rs`, default `127.0.0.1:8080`.

## How is this different from vLLM?

vLLM is a production engine of tens of thousands of lines with real CUDA kernels; forge-infer is a few hundred readable lines with a fake model. They implement the same three core ideas (paged KV-cache, continuous batching, speculative decoding); forge-infer exists to show those ideas standalone. A fuller comparison, including TGI and llama.cpp, is on [Comparisons](Comparisons).

## See also

[Home](Home) for orientation and the full page index.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
