//! # forge-infer
//!
//! A minimal LLM inference server that implements the real serving techniques:
//! a paged KV-cache, continuous batching and speculative decoding.
//!
//! forge-infer is a teaching-grade engine I built to show the systems that make
//! LLM serving fast, with the hard parts implemented for real rather than
//! mocked. The model behind it is a small deterministic transformer so the build
//! stays fast and the tests are reproducible, but the cache allocator, the
//! scheduler and the speculative decoder are the genuine algorithms a production
//! engine uses.
//!
//! ## Module map
//!
//! - [`model`]: the `Model` trait and a deterministic `TinyTransformer`.
//! - [`paged_cache`]: the block-based paged KV-cache allocator.
//! - [`scheduler`]: continuous batching with preemption.
//! - [`speculative`]: the draft-then-verify speculative decoder.
//! - [`engine`]: the loop that joins the scheduler to the model.
//! - [`server`]: the axum HTTP server and OpenAI-compatible endpoint.
//! - [`tokeniser`]: a reversible byte-level tokeniser.

pub mod engine;
pub mod model;
pub mod paged_cache;
pub mod scheduler;
pub mod server;
pub mod speculative;
pub mod tokeniser;

use std::sync::Arc;

/// Build the standard server state used by the binary and integration tests: a
/// deterministic model sized to the tokeniser vocabulary, with sensible cache
/// and batch limits.
pub fn default_state() -> server::AppState {
    let model =
        model::TinyTransformer::new(tokeniser::VOCAB_SIZE, 6).with_eos(tokeniser::EOS_TOKEN);
    server::AppState {
        model: Arc::new(model),
        blocks: 512,
        block_size: 16,
        max_batch_size: 16,
    }
}
