# Model and Tokeniser

This page covers the two modules that sit at the bottom of the stack: the `Model` trait and its deterministic `TinyTransformer` implementation (`src/model.rs`), and the reversible byte-level tokeniser (`src/tokeniser.rs`). Together they are the part of forge-infer that a real deployment would replace. Everything above them, the cache, the scheduler, the decoder, the engine, the server, stays exactly as it is.

## The Model trait

```rust
pub trait Model: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn eos_token(&self) -> TokenId;
    fn forward(&self, context: &[TokenId]) -> StepLogits;
}
```

Four methods, and `forward` is the only one that does work. The `Send + Sync` bound is what lets an `Arc<dyn Model>` be shared across the server's request tasks. `vocab_size` fixes the length of every logits vector. `num_layers` exists so a real backend can tell the cache how many key/value slots to reserve per token; the `TinyTransformer` reports it but does not use it, because it has no real attention layers. `eos_token` is the id the engine compares against to detect end of sequence.

`forward` takes the full context (a slice of token ids) and returns the logits for the position that follows the last context token. The engine calls it once per generated token per sequence. This is the entire contract; the [Writing-a-Model-Backend](Writing-a-Model-Backend) page shows how to implement it against real weights.

## StepLogits

```rust
pub struct StepLogits {
    pub logits: Vec<f32>,
}
```

A `StepLogits` is just the next-token logits vector. It offers two methods that the serving stack depends on:

- `argmax()` returns the greedy next token, breaking ties towards the lower id so the output is fully deterministic. The engine uses this for greedy decoding.
- `prob_of(token)` returns the softmax probability mass on a given token. The speculative decoder uses this to compute `p(t)` and `q(t)` for its acceptance test. The implementation subtracts the max logit before exponentiating, the standard numerically stable softmax, and returns 0.0 for an out-of-range token id rather than panicking.

The test `probabilities_sum_to_one` checks that `prob_of` over the whole vocabulary sums to 1 within 1e-4, which is the property the acceptance test relies on.

## TinyTransformer: a deterministic stand-in

The model behind forge-infer is a fixed-weight, transformer-shaped function whose only job is to be cheap, deterministic and pure. It is not a language model and makes no claim to be. It maps a trailing context window to a peaked logits distribution through a splitmix-style hash.

```rust
fn hash_context(&self, context: &[TokenId]) -> u64 {
    let mut h = self.seed;
    let start = context.len().saturating_sub(8);
    for (pos, &tok) in context[start..].iter().enumerate() {
        h ^= (tok as u64).wrapping_add((0x9E37_79B9 + (pos as u64)) << 6);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 27;
    }
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    h
}
```

The hash runs over the last eight tokens. Limiting the window to a trailing slice is a deliberate echo of attention, which weights recent context most heavily; it also keeps the function fast regardless of how long the sequence grows.

`forward` turns the hash into a peaked distribution. One token (the peak) gets the largest logit, and logits fall off linearly with distance from the peak, clamped at a floor:

```rust
let base = (h % self.vocab as u64) as usize;
// ... draft divergence shifts the peak on a quarter of contexts ...
for (i, l) in logits.iter_mut().enumerate() {
    let d = ((i as i64 - peak as i64).abs() as f32).min(8.0);
    *l = 6.0 - d;
}
```

This shape gives `argmax` a stable winner and gives `prob_of` a realistic-looking softmax for the acceptance test. There is one guard: for contexts shorter than four tokens, the eos logit is forced to negative infinity, so short prompts always produce visible output in tests and benchmarks rather than terminating immediately.

### Why a hash and not a real tiny transformer

This is the central design call of the whole repository and it is argued in full on [Design-Decisions](Design-Decisions). The short version: the speculative acceptance test needs `p(t)/q(t)` to be bit-for-bit reproducible across two model instances, and floating-point attention is not stable enough to assert on. A pure hash is. It also keeps the build to a couple of seconds with no ML dependency tree. The three properties the serving stack needs, all delivered by the hash, are: it is a pure function of the context (so caching is correct to verify), it is deterministic (so the decoder's accept/reject is assertable), and it is cheap (so a benchmark stresses the scheduler, not CUDA).

## The draft model

`TinyTransformer::draft` builds a model that agrees with the base target on most contexts and diverges on a deterministic minority:

```rust
pub fn draft(vocab: usize, layers: usize) -> Self {
    let mut m = Self::new(vocab, layers);
    m.draft_divergence = Some(4);   // diverge when hash % 4 == 0
    m
}
```

When `draft_divergence` is set, `forward` shifts the peak by a fixed offset of 7 whenever `hash % 4 == 0`, that is, on roughly a quarter of contexts. The result is a draft that proposes the same token as the target three times in four and a different token the rest of the time. That is exactly the regime speculative decoding is built for: a cheap proposer that is usually right. The test `draft_and_target_mostly_agree` checks the draft agrees with the target on more than 120 of 200 contexts, confirming it is a useful proposer rather than noise.

## Pinning for tests

```rust
pub fn pin(&mut self, context: Vec<TokenId>, next: TokenId) { ... }
```

`pin` forces a known `(context -> next token)` mapping so an integration test can assert on a specific streamed continuation. It is a test affordance, not part of the serving path. `pinned_context_forces_token` checks it works.

## The tokeniser

```rust
pub const VOCAB_SIZE: usize = 320;
pub const EOS_TOKEN: TokenId = 0;
```

A real engine ships a byte-pair or sentencepiece vocabulary. That machinery is orthogonal to the serving techniques this project is about, so forge-infer uses a reversible byte-level codec: each input byte becomes one token, offset by one so that token 0 stays reserved for eos.

```rust
pub fn encode(text: &str) -> Vec<TokenId> {
    text.bytes().map(|b| b as TokenId + 1).collect()
}
```

Encoding maps byte `b` to token `b + 1`. The vocabulary is 320 wide: 256 byte values plus a small reserved range above them, with token 0 as eos. The 64-token headroom above the byte range exists so the model can occasionally emit a non-byte token without it being mistaken for text.

### Decoding and the streaming fragment

```rust
pub fn decode(tokens: &[TokenId]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len());
    for &t in tokens {
        if t == EOS_TOKEN { continue; }
        if (1..=256).contains(&t) { bytes.push((t - 1) as u8); }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
```

`decode` skips eos, maps tokens in the byte range back to their byte, and ignores out-of-range tokens. `from_utf8_lossy` guarantees the output is always valid UTF-8 even if the byte stream is not, which matters for an HTTP response. `decode_one` does the same for a single token, used by the SSE path where tokens arrive one at a time; a non-byte token renders as the empty string so a streamed fragment is never invalid.

### Data format summary

| Token id range | Meaning | Renders as |
| --- | --- | --- |
| `0` (`EOS_TOKEN`) | end of sequence | skipped in `decode`, empty in `decode_one` |
| `1..=256` | byte `id - 1` | that byte |
| `257..=319` | reserved non-byte | empty string |

The round-trip property, `decode(encode(text)) == text` for any ASCII input, is checked by `round_trips_ascii`, and `decode_one_matches_decode` confirms streaming one token at a time reassembles to the same string as decoding the whole vector.

## What the tests prove

Model: `forward_is_deterministic`, `probabilities_sum_to_one`, `pinned_context_forces_token`, `draft_and_target_mostly_agree`. Tokeniser: `round_trips_ascii`, `eos_is_reserved_and_skipped`, `decode_one_matches_decode`.

## See also

- [Speculative-Decoding](Speculative-Decoding) uses `prob_of` and the draft/target divergence directly.
- [Writing-a-Model-Backend](Writing-a-Model-Backend) for replacing the hash with real weights.
- [API-Reference](API-Reference) for the full signatures.

---
SarmaLinux . sarmalinux.com . [forge-infer repository](https://github.com/sarmakska/forge-infer)
