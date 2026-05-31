# Troubleshooting

Answers to the questions that come up when building, running and extending forge-infer.

## Build

### `cargo` is not found

Make sure the Rust toolchain is on your PATH:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version   # should report 1.96 or newer
```

forge-infer targets Rust 1.96 (set as `rust-version` in `Cargo.toml`). Older toolchains may not have the standard-library methods the code uses, such as `div_ceil` and `is_multiple_of`.

### clippy or rustfmt is missing

CI runs `cargo fmt --check` and `cargo clippy -D warnings`. Install the components locally so you can reproduce CI:

```bash
rustup component add clippy rustfmt
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

### The first build is slow

The dependency tree (tokio, axum, reqwest for tests) compiles on the first build and is then cached. Subsequent builds are fast. The library and binaries themselves are small; the deterministic model means there are no heavy ML crates to compile.

## Running the server

### The server exits immediately or cannot bind

The default bind address is `127.0.0.1:8080`. If that port is in use, choose another:

```bash
FORGE_ADDR=127.0.0.1:9000 cargo run --release --bin forge-infer
```

### A request returns odd-looking text

The model is a deterministic hash-based stand-in, not a language model, so the generated text is a reproducible byte stream rather than meaningful prose. That is expected. The tokeniser is a reversible byte-level codec, so the prompt round-trips exactly and the completion is valid UTF-8. To produce real text, implement the `Model` trait against real weights (see below).

### Streaming returns everything at once

The `/v1/completions` SSE path generates the full completion and then emits one event per token followed by `[DONE]`. Because the model is near-instant, the events arrive together. With a real model the per-token cost spaces them out. The wire format is genuine SSE: use `curl -N` to see the individual `data:` lines.

## Behaviour questions

### Why does a request never finish, or loop?

The engine has a guard of one million iterations in `run_to_completion` to prevent a pathological loop from hanging. If you hit it, check that your `Model::eos_token` and the sequence's `max_new_tokens` are consistent. The deterministic model suppresses eos for very short contexts so that short prompts always produce visible output.

### Why was my large prompt rejected?

A prompt that needs more blocks than the cache holds can never be admitted; `admission_blocks_when_prompt_does_not_fit` documents this. Increase `blocks` or `block_size` in the cache, or in `default_state` for the server. The cache never leaks blocks on a failed admission.

### How does preemption interact with my request?

Under block pressure the scheduler may preempt the least progressed running sequence and resume it later. This is invisible to the client: the request still completes, it just yields the engine for a while. See [Continuous-Batching](Continuous-Batching) for the exact rule.

## Extending forge-infer

### Plugging in a real model

Implement the `Model` trait:

```rust
impl Model for MyBackend {
    fn vocab_size(&self) -> usize { /* ... */ }
    fn num_layers(&self) -> usize { /* ... */ }
    fn eos_token(&self) -> TokenId { /* ... */ }
    fn forward(&self, context: &[TokenId]) -> StepLogits { /* run your weights */ }
}
```

Then build `AppState` with your model instead of `TinyTransformer`. Nothing in the cache, scheduler or decoder needs to change. For real performance you would also keep KV state across `forward` calls rather than recomputing from the full context; the `BlockTable` from `PagedKVCache::block_table` gives you the physical blocks to attend over.

### Tuning concurrency and memory

The knobs are `max_batch_size` (in `SchedulerConfig`) and `num_blocks` plus `block_size` (in `PagedKVCache::new`). Larger batches and more blocks raise concurrency at the cost of memory. See [Paged-KV-Cache](Paged-KV-Cache) and [Continuous-Batching](Continuous-Batching).

## Still stuck

Open an issue at https://github.com/sarmakska/forge-infer/issues with the command you ran, the full output, and your `rustc --version`. For security reports, follow the process in `SECURITY.md` instead of opening a public issue.
