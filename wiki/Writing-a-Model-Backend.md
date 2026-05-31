# Writing a Model Backend

The whole serving stack hangs off one trait. To make forge-infer produce real text, you implement that trait against your own weights and change nothing else. This page is the step-by-step: the contract, a worked skeleton, how to wire the KV-cache through for real performance, the pitfalls, and how to contribute the result back.

## The seam

```rust
pub trait Model: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn eos_token(&self) -> TokenId;
    fn forward(&self, context: &[TokenId]) -> StepLogits;
}
```

`forward` is the only method that does work. It takes the full context (token ids) and returns the logits for the next position as a `StepLogits { logits: Vec<f32> }` of length `vocab_size()`. The engine calls it once per generated token per sequence and reads `argmax()` for greedy decoding; the speculative decoder additionally reads `prob_of(token)` for its acceptance test. The `Send + Sync` bound is what lets your model be shared as an `Arc<dyn Model>` across the server's request tasks.

## A minimal skeleton

```rust
use forge_infer::model::{Model, StepLogits, TokenId};

pub struct MyBackend {
    vocab: usize,
    layers: usize,
    eos: TokenId,
    // your weights, runtime handle, device, etc.
}

impl Model for MyBackend {
    fn vocab_size(&self) -> usize { self.vocab }
    fn num_layers(&self) -> usize { self.layers }
    fn eos_token(&self) -> TokenId { self.eos }

    fn forward(&self, context: &[TokenId]) -> StepLogits {
        // 1. map token ids to your model's input representation
        // 2. run a forward pass
        // 3. return the next-token logits, length == vocab
        let logits: Vec<f32> = run_your_model(context);
        debug_assert_eq!(logits.len(), self.vocab);
        StepLogits { logits }
    }
}
```

That is the entire integration. The cache, scheduler, decoder, engine and server are untouched.

## Wiring it into the server

The server holds an `AppState` with an `Arc<dyn Model>`. Build one with your backend and pass it to `server::router`:

```rust
use forge_infer::server::{self, AppState};
use std::sync::Arc;

let state = AppState {
    model: Arc::new(MyBackend::load("path/to/weights")),
    blocks: 1024,
    block_size: 16,
    max_batch_size: 16,
};
let app = server::router(state);
// serve `app` exactly as src/main.rs does
```

You will likely also bring your model's real tokeniser instead of the byte-level codec in `src/tokeniser.rs`; the codec is a placeholder for the same reason the model is (see [Model-and-Tokeniser](Model-and-Tokeniser)). Encode the prompt with your tokeniser before `submit` and decode the output tokens with it after.

## The performance step: carry KV state across calls

The skeleton above works but is slow, because the engine rebuilds the full context and calls `forward` from scratch every token (see [Engine](Engine)). The `TinyTransformer` does not care, because it is a stateless hash. A real backend should not recompute the whole prefix each step. This is where the paged cache earns its place.

```mermaid
flowchart LR
    ENG["engine calls forward(context)"] --> BT["look up block_table(seq)"]
    BT --> KV["gather physical KV blocks"]
    KV --> ATT["attend new token over cached K/V"]
    ATT --> WR["write new token's K/V into its block"]
    WR --> LOG["return next-token logits"]

    classDef sky fill:#0d1117,stroke:#38bdf8,color:#f5f7fa;
    classDef cyan fill:#0d1117,stroke:#22d3ee,color:#f5f7fa;
    classDef em fill:#0d1117,stroke:#34d399,color:#f5f7fa;
    class ENG sky;
    class BT,KV,ATT cyan;
    class WR,LOG em;
```

The `PagedKVCache` already maintains, per sequence, the ordered list of physical blocks holding its tokens (`PagedKVCache::block_table` returns the `BlockTable`). A real backend would: store its key and value tensors in those physical blocks, gather them for the attention computation, attend the new token over the cached keys and values rather than recomputing them, and write the new token's key and value into the sequence's current block. Each block holds `block_size * num_layers * 2` tensors in a real engine, which is why `num_layers()` is on the trait. Reserving and freeing those blocks is exactly what the scheduler already drives through `append` and `free`; your backend only has to honour the block table the cache hands it. This is the work the [Roadmap-and-Limitations](Roadmap-and-Limitations) calls out as the gap between the teaching engine and a real one.

## Keeping speculative decoding exact

If you want speculative decoding with your backend, the acceptance test relies on `prob_of(token)` returning a faithful softmax probability from both the draft and the target. As long as your `forward` returns real logits, `StepLogits::prob_of` computes a numerically stable softmax for you. The exactness guarantee holds in expectation for any draft/target pair sampled correctly. The one caveat from [Design-Decisions](Design-Decisions): with floating-point models you cannot assert bit-identical output across instances the way the deterministic model's test does; the guarantee becomes distributional, not token-for-token. That is a property of real models, not of forge-infer.

## Pitfalls

- **Wrong logits length.** `StepLogits.logits` must have exactly `vocab_size()` entries. `argmax` and `prob_of` index into it; a short vector silently changes results, a long one wastes work. The `debug_assert_eq!` in the skeleton catches it in debug builds.
- **An eos that the suppression logic fights.** The `TinyTransformer` suppresses eos for very short contexts so tests always see output; your real model will not, and should not. Just return your real eos logit. If a sequence terminates immediately, check your `eos_token()` matches what your model actually emits.
- **Blocking the async runtime.** A heavy synchronous `forward` called from an async handler will stall the tokio worker. Run the engine loop on a blocking task or a dedicated thread. This matters more once you close the server-side single-engine gap.
- **Non-`Send` runtime handles.** The trait requires `Send + Sync`. If your inference library uses a non-`Send` handle, wrap it so the bound holds (a mutex or a per-thread instance), or the `Arc<dyn Model>` will not compile.

## Contributing back

forge-infer keeps the default build fast and dependency-light on purpose, so a real backend belongs behind an optional feature flag rather than as a default dependency. The roadmap names a `candle` feature flag as the intended shape: the backend compiles only when the feature is on, the default build stays a couple of seconds with no ML crates. If you build a backend worth sharing:

1. Put it behind a Cargo feature so the default build is unaffected.
2. Keep the `Model` trait the only contact point; do not leak your backend's types into the cache, scheduler or decoder.
3. Add a test that your `forward` returns a `vocab_size()`-length vector and a sane `argmax`.
4. Follow the contribution norms in the repository: small, logically ordered commits with specific messages, `cargo fmt` clean, `cargo clippy -D warnings` clean, `cargo test` green.

Open a pull request at the [repository](https://github.com/sarmakska/forge-infer). For anything that touches the serving algorithms rather than just the model seam, open an issue first so we can agree the shape; the project guards its surface area deliberately (see [Roadmap-and-Limitations](Roadmap-and-Limitations)).

## See also

- [Model-and-Tokeniser](Model-and-Tokeniser) for the trait and the deterministic stand-in it replaces.
- [Engine](Engine) for why the default path recomputes and where to hook stateful KV.
- [API-Reference](API-Reference) for the exact signatures.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
