//! The model abstraction that the serving stack runs against.
//!
//! forge-infer is a teaching-grade serving engine. The interesting work lives in
//! the scheduler, the paged KV-cache and the speculative decoder, not in the
//! matrix multiplies. To keep the build fast and the tests reproducible I run
//! the whole stack against a small fixed-weight transformer that is fully
//! deterministic. Every component that a real engine has (a key/value cache, a
//! batched forward pass, a logits vector per position) is present, it is simply
//! sized so that the build stays under a couple of seconds and the tests never
//! flake.

use std::collections::HashMap;

/// A token identifier. The vocabulary here is tiny on purpose.
pub type TokenId = u32;

/// The result of a single forward step for one sequence: the next-token logits.
#[derive(Clone, Debug)]
pub struct StepLogits {
    pub logits: Vec<f32>,
}

impl StepLogits {
    /// The greedy argmax over the logits. Ties break towards the lower id so the
    /// output is fully deterministic.
    pub fn argmax(&self) -> TokenId {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in self.logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as TokenId
    }

    /// The probability mass assigned to `token` after a softmax over the logits.
    /// The speculative decoder uses this for its acceptance test.
    pub fn prob_of(&self, token: TokenId) -> f32 {
        let max = self
            .logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for &v in &self.logits {
            denom += (v - max).exp();
        }
        let idx = token as usize;
        if idx >= self.logits.len() {
            return 0.0;
        }
        (self.logits[idx] - max).exp() / denom
    }
}

/// The contract every model implements. The serving stack only ever talks to
/// this trait, which is what lets the speculative decoder pair a cheap draft
/// model with an expensive target model without either knowing about the other.
pub trait Model: Send + Sync {
    /// The size of the vocabulary. Logits vectors always have this length.
    fn vocab_size(&self) -> usize;

    /// The number of transformer layers. The KV-cache reserves one key/value
    /// slot per layer per token.
    fn num_layers(&self) -> usize;

    /// The id used to signal end of sequence.
    fn eos_token(&self) -> TokenId;

    /// Run a forward pass over a context and return the logits for the position
    /// that follows the last context token. This is the function the engine
    /// calls once per generated token per sequence.
    fn forward(&self, context: &[TokenId]) -> StepLogits;
}

/// A small fixed-weight transformer-shaped model.
///
/// The map from a context to the next token is a deterministic hash of the
/// trailing context window. That gives us three properties the serving stack
/// needs to be testable:
///
/// 1. It is a pure function of the context, so caching is correct to verify.
/// 2. It is deterministic, so the speculative decoder either accepts or rejects
///    a draft token in a way a test can assert.
/// 3. It is cheap, so a benchmark can drive thousands of tokens per second and
///    we can reason about the scheduler rather than about CUDA.
///
/// A draft variant built with [`TinyTransformer::draft`] shares the target's
/// hashing but shifts its peak on a deterministic minority of contexts, so a
/// draft model and a target model agree on most tokens and disagree on some,
/// which is the realistic regime for speculative decoding.
pub struct TinyTransformer {
    vocab: usize,
    layers: usize,
    eos: TokenId,
    seed: u64,
    /// When set, this model behaves as a draft: it predicts the same token as a
    /// target built from the base seed on most contexts, and diverges on a
    /// minority of them. That is the realistic regime for speculative decoding,
    /// where the draft is right most of the time but not always.
    draft_divergence: Option<u8>,
    /// Forces a known continuation for a known prompt so integration tests can
    /// assert on streamed output.
    pinned: HashMap<Vec<TokenId>, TokenId>,
}

impl TinyTransformer {
    pub fn new(vocab: usize, layers: usize) -> Self {
        Self {
            vocab,
            layers,
            eos: 0,
            seed: 0x9E37_79B9_7F4A_7C15,
            draft_divergence: None,
            pinned: HashMap::new(),
        }
    }

    /// A draft model for the target built with [`TinyTransformer::new`] using
    /// the same vocabulary. It agrees with that target on the majority of
    /// contexts and diverges on roughly one in four, which is exactly the case
    /// speculative decoding is built to exploit: a cheap proposer that is right
    /// most of the time.
    pub fn draft(vocab: usize, layers: usize) -> Self {
        let mut m = Self::new(vocab, layers);
        // Diverge whenever (hash mod 4) == 0, i.e. about a quarter of contexts.
        m.draft_divergence = Some(4);
        m
    }

    pub fn with_eos(mut self, eos: TokenId) -> Self {
        self.eos = eos;
        self
    }

    /// Pin a `(context -> next token)` mapping. Used by tests that need a known
    /// deterministic continuation.
    pub fn pin(&mut self, context: Vec<TokenId>, next: TokenId) {
        self.pinned.insert(context, next);
    }

    fn hash_context(&self, context: &[TokenId]) -> u64 {
        // A small splitmix-style mix over a trailing window. The window keeps
        // the function sensitive to recent tokens, which mirrors how attention
        // weights recent context most heavily.
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
}

impl Model for TinyTransformer {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn num_layers(&self) -> usize {
        self.layers
    }

    fn eos_token(&self) -> TokenId {
        self.eos
    }

    fn forward(&self, context: &[TokenId]) -> StepLogits {
        let mut logits = vec![0.0f32; self.vocab];
        let h = self.hash_context(context);

        // Build a peaked distribution: one token gets a large logit, its
        // neighbours get a smaller share. This makes argmax stable and gives the
        // softmax a realistic shape for the acceptance test.
        let peak = if let Some(&pinned) = self.pinned.get(context) {
            pinned as usize % self.vocab
        } else {
            let base = (h % self.vocab as u64) as usize;
            match self.draft_divergence {
                // A draft diverges from the target on a deterministic minority
                // of contexts, shifting its peak by a fixed offset.
                Some(period) if h.is_multiple_of(period as u64) => (base + 7) % self.vocab,
                _ => base,
            }
        };

        for (i, l) in logits.iter_mut().enumerate() {
            let d = ((i as i64 - peak as i64).abs() as f32).min(8.0);
            *l = 6.0 - d;
        }
        // Never let the model emit token 0 (eos) unless the context is long, so
        // short prompts always produce visible output in tests and benchmarks.
        if context.len() < 4 {
            logits[self.eos as usize] = f32::NEG_INFINITY;
        }
        StepLogits { logits }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_is_deterministic() {
        let m = TinyTransformer::new(64, 4);
        let ctx = vec![1, 2, 3, 4, 5];
        let a = m.forward(&ctx).argmax();
        let b = m.forward(&ctx).argmax();
        assert_eq!(a, b, "the same context must produce the same token");
    }

    #[test]
    fn probabilities_sum_to_one() {
        let m = TinyTransformer::new(32, 2);
        let logits = m.forward(&[7, 8, 9]);
        let total: f32 = (0..32).map(|t| logits.prob_of(t)).sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "softmax must normalise, got {total}"
        );
    }

    #[test]
    fn pinned_context_forces_token() {
        let mut m = TinyTransformer::new(64, 4);
        m.pin(vec![10, 11, 12, 13], 42);
        assert_eq!(m.forward(&[10, 11, 12, 13]).argmax(), 42);
    }

    #[test]
    fn draft_and_target_mostly_agree() {
        let target = TinyTransformer::new(64, 4);
        let draft = TinyTransformer::draft(64, 4);
        let mut agree = 0;
        for s in 0..200u32 {
            let ctx: Vec<TokenId> = (s..s + 6).collect();
            if target.forward(&ctx).argmax() == draft.forward(&ctx).argmax() {
                agree += 1;
            }
        }
        // The draft must be a useful proposer: it should agree with the target
        // on a clear majority of contexts, which is what makes speculation pay.
        assert!(
            agree > 120,
            "draft agreed on only {agree}/200 contexts, too weak a proposer"
        );
    }
}
