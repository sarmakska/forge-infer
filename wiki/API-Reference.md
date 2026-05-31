# API Reference

The complete public surface of forge-infer, both the Rust library (the `forge_infer` crate) and the HTTP API. Symbols are grouped by module and link in spirit to the subsystem pages that explain them in depth. Signatures here are reproduced from the source in `src/`; treat the source as canonical if anything drifts.

## Crate layout

```
forge_infer (lib)        src/lib.rs
  ::model                src/model.rs
  ::tokeniser            src/tokeniser.rs
  ::paged_cache          src/paged_cache.rs
  ::scheduler            src/scheduler.rs
  ::speculative          src/speculative.rs
  ::engine               src/engine.rs
  ::server               src/server.rs

forge-infer (bin)        src/main.rs      the server
forge-bench (bin)        src/bin/bench.rs the benchmark
```

`default_state() -> server::AppState` lives at the crate root (`src/lib.rs`) and builds the standard server state used by the binary and the integration tests.

## model

```rust
pub type TokenId = u32;

pub struct StepLogits { pub logits: Vec<f32> }
impl StepLogits {
    pub fn argmax(&self) -> TokenId;
    pub fn prob_of(&self, token: TokenId) -> f32;
}

pub trait Model: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn eos_token(&self) -> TokenId;
    fn forward(&self, context: &[TokenId]) -> StepLogits;
}

pub struct TinyTransformer { /* private */ }
impl TinyTransformer {
    pub fn new(vocab: usize, layers: usize) -> Self;
    pub fn draft(vocab: usize, layers: usize) -> Self;
    pub fn with_eos(self, eos: TokenId) -> Self;
    pub fn pin(&mut self, context: Vec<TokenId>, next: TokenId);
}
impl Model for TinyTransformer { /* ... */ }
```

See [Model-and-Tokeniser](Model-and-Tokeniser). `argmax` breaks ties towards the lower id. `prob_of` returns a numerically stable softmax probability, 0.0 for out-of-range ids. `draft` builds a proposer that diverges from the base target on a deterministic quarter of contexts. `with_eos` is a builder-style setter chained off `new`/`draft`.

## tokeniser

```rust
pub const VOCAB_SIZE: usize = 320;
pub const EOS_TOKEN: TokenId = 0;

pub fn encode(text: &str) -> Vec<TokenId>;
pub fn decode(tokens: &[TokenId]) -> String;
pub fn decode_one(token: TokenId) -> String;
```

Byte `b` encodes to token `b + 1`; token 0 is eos; tokens `257..=319` are reserved non-byte ids. `decode` is lossy-UTF-8 safe. See [Model-and-Tokeniser](Model-and-Tokeniser).

## paged_cache

```rust
pub type SeqId = u64;
pub type BlockId = usize;

pub enum CacheError {
    OutOfBlocks { needed: usize, free: usize },
    UnknownSequence(SeqId),
}

pub struct BlockTable { pub blocks: Vec<BlockId>, pub num_tokens: usize }
impl BlockTable {
    pub fn capacity(&self, block_size: usize) -> usize;
    pub fn slack(&self, block_size: usize) -> usize;
}

pub struct PagedKVCache { /* private */ }
impl PagedKVCache {
    pub fn new(num_blocks: usize, block_size: usize) -> Self;   // asserts block_size > 0
    pub fn block_size(&self) -> usize;
    pub fn total_blocks(&self) -> usize;
    pub fn free_blocks(&self) -> usize;
    pub fn used_blocks(&self) -> usize;
    pub fn utilisation(&self) -> f32;
    pub fn admit(&mut self, seq: SeqId);
    pub fn contains(&self, seq: SeqId) -> bool;
    pub fn blocks_needed_for(&self, seq: SeqId, extra_tokens: usize) -> usize;
    pub fn append(&mut self, seq: SeqId, count: usize) -> Result<(), CacheError>;
    pub fn free(&mut self, seq: SeqId);
    pub fn block_table(&self, seq: SeqId) -> Option<&BlockTable>;
    pub fn seq_len(&self, seq: SeqId) -> usize;
    pub fn internal_fragmentation(&self) -> usize;
    pub fn evict_largest(&mut self) -> Option<SeqId>;
}
```

`append` is transactional: on `OutOfBlocks` the sequence is left untouched. `free` is idempotent. See [Paged-KV-Cache](Paged-KV-Cache).

## scheduler

```rust
pub enum SeqState { Waiting, Running, Preempted, Finished }

pub struct Sequence {
    pub id: SeqId,
    pub prompt: Vec<TokenId>,
    pub output: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub state: SeqState,
}
impl Sequence {
    pub fn new(id: SeqId, prompt: Vec<TokenId>, max_new_tokens: usize) -> Self;
    pub fn total_len(&self) -> usize;
    pub fn is_complete(&self) -> bool;
}

pub struct StepPlan {
    pub admitted: Vec<SeqId>,
    pub decode_batch: Vec<SeqId>,
    pub preempted: Vec<SeqId>,
    pub finished: Vec<SeqId>,
}
impl StepPlan { pub fn is_idle(&self) -> bool }

pub struct SchedulerConfig { pub max_batch_size: usize }   // Default: 8

pub struct Scheduler { /* private */ }
impl Scheduler {
    pub fn new(config: SchedulerConfig, cache: PagedKVCache) -> Self;
    pub fn cache(&self) -> &PagedKVCache;
    pub fn cache_mut(&mut self) -> &mut PagedKVCache;
    pub fn submit(&mut self, seq: Sequence);
    pub fn waiting_len(&self) -> usize;
    pub fn running_len(&self) -> usize;
    pub fn has_work(&self) -> bool;
    pub fn running_seq_mut(&mut self, id: SeqId) -> Option<&mut Sequence>;
    pub fn schedule(&mut self) -> StepPlan;
    pub fn push_token(&mut self, id: SeqId, token: TokenId, is_eos: bool);
    pub fn take_finished(&mut self) -> Vec<SeqId>;   // see note below
}
```

`schedule` mutates the cache and queues and returns its decisions; it never calls the model. `push_token` records a generated token and, on eos, caps the limit so the sequence retires next step. `take_finished` returns an empty vector by design; callers read `StepPlan::finished` instead, and the method is kept only for symmetry in the engine loop. See [Continuous-Batching](Continuous-Batching).

## speculative

```rust
pub struct SpeculationResult {
    pub tokens: Vec<TokenId>,
    pub accepted: usize,
    pub drafted: usize,
}
impl SpeculationResult { pub fn acceptance_rate(&self) -> f32 }

pub struct SpeculativeDecoder<'a> { /* private */ }
impl<'a> SpeculativeDecoder<'a> {
    pub fn new(draft: &'a dyn Model, target: &'a dyn Model, lookahead: usize) -> Self;
    pub fn step(&self, context: &[TokenId]) -> SpeculationResult;
    pub fn generate(&self, prompt: &[TokenId], max_new: usize) -> (Vec<TokenId>, f32);
}
```

`new` asserts `lookahead >= 1` and that draft and target share a vocabulary. `step` verifies one drafted run and always emits at least one token. `generate` runs rounds to `max_new`, returning the tokens and the aggregate acceptance rate. See [Speculative-Decoding](Speculative-Decoding).

## engine

```rust
pub struct GenerationOutput { pub id: u64, pub tokens: Vec<TokenId> }

pub struct Engine { /* private */ }
impl Engine {
    pub fn new(model: Arc<dyn Model>, config: SchedulerConfig, cache: PagedKVCache) -> Self;
    pub fn scheduler_mut(&mut self) -> &mut Scheduler;
    pub fn submit(&mut self, id: u64, prompt: Vec<TokenId>, max_new_tokens: usize);
    pub fn step(&mut self) -> Vec<(u64, TokenId, bool)>;
    pub fn scheduler_mut_run(&mut self, _prompt_len: usize, _max_new: usize) -> Vec<TokenId>;
    pub fn run_to_completion(&mut self) -> Vec<GenerationOutput>;
}
```

`step` returns `(id, token, is_eos)` triples for the tokens produced this iteration. `run_to_completion` drives until idle and returns per-request outputs sorted by id, guarded at one million iterations. See [Engine](Engine).

## server

```rust
pub struct AppState {
    pub model: Arc<dyn Model>,
    pub blocks: usize,
    pub block_size: usize,
    pub max_batch_size: usize,
}

pub fn router(state: AppState) -> Router;
```

`router` builds the axum router with `/healthz`, `/generate` and `/v1/completions`. Request and response JSON types (`GenerateRequest`, `CompletionRequest` and friends) are private to the module; their wire shape is documented on [HTTP-Server](HTTP-Server).

## HTTP API

| Method | Path | Request body | Response |
| --- | --- | --- | --- |
| `GET` | `/healthz` | none | `{"status":"ok"}` |
| `POST` | `/generate` | `{prompt: string, max_tokens?: int=32}` | `{text, prompt_tokens, completion_tokens}` |
| `POST` | `/v1/completions` | `{prompt: string, max_tokens?: int=32, stream?: bool=false, model?: string="forge-infer"}` | OpenAI completion JSON, or `text/event-stream` if `stream` |

Streaming events carry `{id, object, model, choices:[{text, index, finish_reason}]}` with `finish_reason: null` per token, terminated by `data: [DONE]`.

## Environment

| Variable | Default | Effect |
| --- | --- | --- |
| `FORGE_ADDR` | `127.0.0.1:8080` | server bind address (`src/main.rs`) |
| `RUST_LOG` | `forge_infer=info` | tracing filter |

## See also

[Configuration-and-Tuning](Configuration-and-Tuning) for the knobs and how to set them, and [Writing-a-Model-Backend](Writing-a-Model-Backend) for implementing `Model`.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
