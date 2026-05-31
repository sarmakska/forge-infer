//! forge-bench: a throughput and latency benchmark for the engine.
//!
//! It drives the engine three ways and prints a comparison table:
//!
//! 1. Sequential decoding, one request at a time (the static-batching baseline).
//! 2. Continuous batching, many requests sharing the engine loop.
//! 3. Speculative decoding for a single stream.
//!
//! Because the model is a cheap deterministic function, the absolute numbers
//! reflect scheduling overhead and algorithmic structure rather than GPU
//! kernels. That is the point: the benchmark isolates the serving techniques.

use forge_infer::engine::Engine;
use forge_infer::model::{Model, TinyTransformer};
use forge_infer::paged_cache::PagedKVCache;
use forge_infer::scheduler::SchedulerConfig;
use forge_infer::speculative::SpeculativeDecoder;
use forge_infer::tokeniser::{EOS_TOKEN, VOCAB_SIZE};
use std::sync::Arc;
use std::time::Instant;

const NUM_REQUESTS: usize = 64;
const PROMPT_LEN: usize = 16;
const MAX_NEW: usize = 64;

fn model() -> Arc<dyn Model> {
    Arc::new(TinyTransformer::new(VOCAB_SIZE, 6).with_eos(EOS_TOKEN))
}

fn prompt(i: usize) -> Vec<u32> {
    (0..PROMPT_LEN)
        .map(|j| ((i * 7 + j * 3) % 200 + 1) as u32)
        .collect()
}

/// Sequential: a fresh engine per request, batch size 1. This mimics the
/// static-batching baseline where requests do not share decode steps.
fn run_sequential() -> (usize, f64) {
    let model = model();
    let start = Instant::now();
    let mut total_tokens = 0;
    for i in 0..NUM_REQUESTS {
        let mut eng = Engine::new(
            model.clone(),
            SchedulerConfig { max_batch_size: 1 },
            PagedKVCache::new(256, 16),
        );
        eng.submit(i as u64, prompt(i), MAX_NEW);
        for o in eng.run_to_completion() {
            total_tokens += o.tokens.len();
        }
    }
    (total_tokens, start.elapsed().as_secs_f64())
}

/// Continuous batching: one engine, all requests submitted up front, batch
/// size 16. Requests share decode iterations and the engine stays saturated.
fn run_continuous() -> (usize, f64) {
    let model = model();
    let start = Instant::now();
    let mut eng = Engine::new(
        model,
        SchedulerConfig { max_batch_size: 16 },
        PagedKVCache::new(2048, 16),
    );
    for i in 0..NUM_REQUESTS {
        eng.submit(i as u64, prompt(i), MAX_NEW);
    }
    let mut total_tokens = 0;
    for o in eng.run_to_completion() {
        total_tokens += o.tokens.len();
    }
    (total_tokens, start.elapsed().as_secs_f64())
}

/// Speculative decoding for a single stream, reporting the acceptance rate.
fn run_speculative() -> (usize, f64, f32) {
    let target = TinyTransformer::new(VOCAB_SIZE, 6).with_eos(EOS_TOKEN);
    let draft = TinyTransformer::draft(VOCAB_SIZE, 6).with_eos(EOS_TOKEN);
    let dec = SpeculativeDecoder::new(&draft, &target, 4);
    let start = Instant::now();
    let mut total = 0;
    let mut rate_sum = 0.0f32;
    for i in 0..NUM_REQUESTS {
        let (toks, rate) = dec.generate(&prompt(i), MAX_NEW);
        total += toks.len();
        rate_sum += rate;
    }
    (
        total,
        start.elapsed().as_secs_f64(),
        rate_sum / NUM_REQUESTS as f32,
    )
}

fn main() {
    println!("forge-infer benchmark");
    println!("requests={NUM_REQUESTS} prompt_len={PROMPT_LEN} max_new_tokens={MAX_NEW}");
    println!();

    let (seq_tok, seq_s) = run_sequential();
    let (cont_tok, cont_s) = run_continuous();
    let (spec_tok, spec_s, spec_rate) = run_speculative();

    let row = |name: &str, tokens: usize, secs: f64, extra: String| {
        let tps = tokens as f64 / secs.max(1e-9);
        let lat_ms = secs / NUM_REQUESTS as f64 * 1000.0;
        println!("{name:<22} {tokens:>8} {tps:>14.0} {lat_ms:>14.3}  {extra}");
    };

    let header = format!(
        "{:<22} {:>8} {:>14} {:>14}  {}",
        "strategy", "tokens", "tokens/sec", "ms/request", "notes"
    );
    println!("{header}");
    println!("{}", "-".repeat(86));
    row("sequential", seq_tok, seq_s, "batch=1 baseline".to_string());
    row(
        "continuous-batching",
        cont_tok,
        cont_s,
        "batch=16".to_string(),
    );
    row(
        "speculative",
        spec_tok,
        spec_s,
        format!("acceptance={:.0}%", spec_rate * 100.0),
    );

    println!();
    let speedup = (cont_tok as f64 / cont_s) / (seq_tok as f64 / seq_s).max(1e-9);
    println!("continuous batching throughput vs sequential: {speedup:.2}x");
}
