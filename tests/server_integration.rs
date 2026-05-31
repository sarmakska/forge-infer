//! End-to-end integration tests that boot the real axum server on an ephemeral
//! port and drive it over HTTP, including the streaming SSE path.

use forge_infer::{default_state, server};
use std::net::SocketAddr;
use tokio::net::TcpListener;

/// Boot the server on a random free port and return its address. The server
/// task runs for the life of the test process.
async fn boot() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = server::router(default_state());
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Give the accept loop a moment to be ready.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    addr
}

#[tokio::test]
async fn healthz_reports_ok() {
    let addr = boot().await;
    let body: serde_json::Value = reqwest::get(format!("http://{addr}/healthz"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn generate_returns_a_completion() {
    let addr = boot().await;
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(format!("http://{addr}/generate"))
        .json(&serde_json::json!({ "prompt": "hello forge", "max_tokens": 24 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(resp["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(resp["completion_tokens"].as_u64().unwrap() > 0);
    assert!(resp["text"].is_string());
}

#[tokio::test]
async fn openai_completions_non_streaming() {
    let addr = boot().await;
    let client = reqwest::Client::new();
    let resp: serde_json::Value = client
        .post(format!("http://{addr}/v1/completions"))
        .json(&serde_json::json!({
            "model": "forge-infer",
            "prompt": "the quick brown fox",
            "max_tokens": 16
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert_eq!(resp["object"], "text_completion");
    assert!(resp["choices"][0]["text"].is_string());
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn openai_completions_streams_sse() {
    let addr = boot().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/completions"))
        .json(&serde_json::json!({
            "model": "forge-infer",
            "prompt": "stream me",
            "max_tokens": 20,
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "expected SSE, got {content_type}"
    );

    let body = resp.text().await.unwrap();
    // The stream must carry at least one data event and end with the sentinel.
    let data_lines: Vec<&str> = body.lines().filter(|l| l.starts_with("data:")).collect();
    assert!(
        data_lines.len() >= 2,
        "expected several SSE events, body: {body}"
    );
    assert!(
        data_lines.iter().any(|l| l.contains("[DONE]")),
        "stream must terminate with [DONE], body: {body}"
    );
    // The non-sentinel events must be valid OpenAI-shaped JSON chunks.
    let first = data_lines[0].trim_start_matches("data:").trim();
    let chunk: serde_json::Value = serde_json::from_str(first).unwrap();
    assert_eq!(chunk["object"], "text_completion");
    assert!(chunk["choices"][0]["text"].is_string());
}
