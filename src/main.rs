//! The forge-infer server binary.
//!
//! Boots the axum server with the default deterministic model and serves the
//! `/generate`, `/v1/completions` and `/healthz` endpoints. The bind address is
//! read from `FORGE_ADDR`, defaulting to `127.0.0.1:8080`.

use forge_infer::{default_state, server};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "forge_infer=info".into()),
        )
        .init();

    let addr: SocketAddr = std::env::var("FORGE_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
        .parse()?;

    let app = server::router(default_state());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("forge-infer listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}
