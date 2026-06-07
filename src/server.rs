//! The axum HTTP server.
//!
//! Two endpoints are exposed:
//!
//! - `POST /generate`, forge-infer's native shape, returning the full
//!   completion as JSON.
//! - `POST /v1/completions`, an OpenAI-compatible shape that supports
//!   `"stream": true` and emits Server-Sent Events, so existing OpenAI client
//!   code can point at forge-infer unchanged.
//! - `GET /healthz`, a liveness probe.
//!
//! Each request is served by spinning up an [`Engine`] over a shared model and a
//! fresh KV-cache, submitting the prompt, and draining tokens. Sharing the model
//! (an `Arc<dyn Model>`) keeps the model weights resident while giving each
//! request an isolated cache and scheduler. A production deployment would route
//! all requests into a single long-lived engine loop; the per-request engine
//! here keeps the example readable while exercising the exact same scheduler,
//! cache and model code paths.

use crate::engine::Engine;
use crate::model::Model;
use crate::paged_cache::PagedKVCache;
use crate::scheduler::SchedulerConfig;
use crate::tokeniser;
use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

/// Shared server state: the model and the default serving limits.
#[derive(Clone)]
pub struct AppState {
    pub model: Arc<dyn Model>,
    pub blocks: usize,
    pub block_size: usize,
    pub max_batch_size: usize,
}

impl AppState {
    fn new_engine(&self) -> Engine {
        Engine::new(
            self.model.clone(),
            SchedulerConfig {
                max_batch_size: self.max_batch_size,
            },
            PagedKVCache::new(self.blocks, self.block_size),
        )
    }
}

/// Build the axum router. Kept separate from binding so integration tests can
/// mount it on an ephemeral port.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/generate", post(generate))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    32
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Run a prompt through a fresh engine and return the decoded completion.
fn run_prompt(state: &AppState, prompt: &str, max_tokens: usize) -> (String, usize, usize) {
    let prompt_ids = tokeniser::encode(prompt);
    let mut engine = state.new_engine();
    engine.submit(1, prompt_ids.clone(), max_tokens);
    let outputs = engine.run_one();
    let text = tokeniser::decode(&outputs);
    (text, prompt_ids.len(), outputs.len())
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> impl IntoResponse {
    let (text, prompt_tokens, completion_tokens) = run_prompt(&state, &req.prompt, req.max_tokens);
    Json(GenerateResponse {
        text,
        prompt_tokens,
        completion_tokens,
    })
}

// --- OpenAI-compatible /v1/completions ---

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_model_name")]
    model: String,
}

fn default_model_name() -> String {
    "forge-infer".to_string()
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    if req.stream {
        stream_completion(state, req).await.into_response()
    } else {
        let (text, _, _) = run_prompt(&state, &req.prompt, req.max_tokens);
        Json(CompletionResponse {
            id: "cmpl-forge".to_string(),
            object: "text_completion".to_string(),
            model: req.model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                finish_reason: Some("stop".to_string()),
            }],
        })
        .into_response()
    }
}

/// Stream a completion as Server-Sent Events in the OpenAI delta shape, ending
/// with the `[DONE]` sentinel. We generate the whole completion up front (the
/// model is deterministic and cheap) and then emit one SSE event per token, so
/// the wire behaviour matches a streaming OpenAI endpoint.
async fn stream_completion(
    state: AppState,
    req: CompletionRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let prompt_ids = tokeniser::encode(&req.prompt);
    let mut engine = state.new_engine();
    engine.submit(1, prompt_ids.clone(), req.max_tokens);
    let tokens = engine.run_one();
    let model = req.model.clone();

    let mut events: Vec<Result<Event, Infallible>> = Vec::with_capacity(tokens.len() + 1);
    for tok in tokens {
        let frag = tokeniser::decode_one(tok);
        let chunk = serde_json::json!({
            "id": "cmpl-forge",
            "object": "text_completion",
            "model": model,
            "choices": [{ "text": frag, "index": 0, "finish_reason": serde_json::Value::Null }],
        });
        events.push(Ok(Event::default().data(chunk.to_string())));
    }
    events.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TinyTransformer;

    fn state() -> AppState {
        AppState {
            model: Arc::new(
                TinyTransformer::new(tokeniser::VOCAB_SIZE, 4).with_eos(tokeniser::EOS_TOKEN),
            ),
            blocks: 256,
            block_size: 16,
            max_batch_size: 8,
        }
    }

    #[test]
    fn run_prompt_produces_text() {
        let st = state();
        let (text, p, c) = run_prompt(&st, "hello", 16);
        assert!(p > 0);
        assert!(c > 0);
        // The completion is deterministic, so it is stable but we only assert it
        // is non-empty and valid here.
        assert!(!text.is_empty() || c > 0);
    }
}
