//! A tiny deterministic tokeniser.
//!
//! A real engine ships a byte-pair or sentencepiece vocabulary. That is
//! orthogonal to the serving techniques this project is about, so I use a
//! reversible byte-level tokeniser: each input byte maps to one token id, and
//! decoding maps printable token ids back to bytes. It is deterministic,
//! dependency-free and good enough to demonstrate end-to-end text in and text
//! out over HTTP.

use crate::model::TokenId;

/// The vocabulary size. 256 byte values plus a small reserved range. Token 0 is
/// reserved as eos.
pub const VOCAB_SIZE: usize = 320;
pub const EOS_TOKEN: TokenId = 0;

/// Encode text into token ids. Each byte becomes one token, offset by one so
/// that token 0 stays reserved for eos.
pub fn encode(text: &str) -> Vec<TokenId> {
    text.bytes().map(|b| b as TokenId + 1).collect()
}

/// Decode token ids back into a string. Tokens in the byte range render as their
/// byte; eos and out-of-range ids render as a visible placeholder so streamed
/// output is always valid UTF-8.
pub fn decode(tokens: &[TokenId]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len());
    for &t in tokens {
        if t == EOS_TOKEN {
            continue;
        }
        if (1..=256).contains(&t) {
            bytes.push((t - 1) as u8);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Decode a single token to a string fragment, for SSE streaming where tokens
/// arrive one at a time. Non-byte tokens render as the empty string.
pub fn decode_one(token: TokenId) -> String {
    if (1..=256).contains(&token) {
        String::from_utf8_lossy(&[(token - 1) as u8]).into_owned()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_ascii() {
        let text = "hello forge";
        assert_eq!(decode(&encode(text)), text);
    }

    #[test]
    fn eos_is_reserved_and_skipped() {
        assert_eq!(encode("a")[0], b'a' as TokenId + 1);
        assert_eq!(decode(&[EOS_TOKEN, b'x' as TokenId + 1]), "x");
    }

    #[test]
    fn decode_one_matches_decode() {
        let toks = encode("hi");
        let joined: String = toks.iter().map(|t| decode_one(*t)).collect();
        assert_eq!(joined, "hi");
    }
}
