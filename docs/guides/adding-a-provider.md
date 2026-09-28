# Adding an LLM Provider

Providers implement the `LLMProvider` trait from `src/core/provider.rs`.

## Step 1: Create the module

```rust
// src/llm/openai.rs
use async_trait::async_trait;
use reqwest::Client;
use crate::core::{AgentError, LLMProvider, LLMResponse, Message, ToolDef};

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { client: Client::new(), api_key, model }
    }
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError> {
        // POST to OpenAI /v1/chat/completions
        // Parse tool_calls from response
        // Return LLMResponse
        todo!()
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    // Optional: both default to a non-streaming shim that wraps `chat`.
    // Override `chat_stream` to stream, and `supports_streaming` to report true,
    // so `LLMNode` uses your streaming path.
}
```

## Step 2: Register the module

```rust
// src/llm/mod.rs
pub mod ollama;
pub mod openai;
```

## Step 3: Use in main.rs

Add a branch to `create_provider()` in `src/main.rs`, and the matching env vars to
`Config` in `src/config.rs`:

```rust
fn create_provider(config: &Config) -> Result<Box<dyn LLMProvider>, AgentError> {
    match config.llm_provider.as_str() {
        "openai" => {
            let api_key = config.openai_api_key.as_ref().ok_or_else(|| {
                AgentError::LLMError("OPENAI_API_KEY must be set".to_string())
            })?;
            Ok(Box::new(OpenAIProvider::new(api_key.clone(), config.openai_model.clone())))
        }
        _ => Ok(Box::new(OllamaProvider::new(config.ollama_host.clone(), config.ollama_model.clone()))),
    }
}
```

Return a `Result` rather than panicking so a missing credential produces a readable
startup error instead of a backtrace.

## Key Points

- The `chat()` method receives messages (including system prompt) and tool definitions
- Return `LLMResponse` with either `content` (final answer) or `tool_calls` (function calling)
- Errors must be wrapped in `AgentError::LLMError(String)`
- Providers are `Send + Sync` — no interior mutability unless behind a lock
- `chat_stream()` returns a `BoxStream` of `StreamingChunk`; accumulate `content` and
  take `tool_calls` from the final `done: true` chunk
- Test with mock providers — see the `tests` module in `src/agent/mod.rs`
  (`MockProvider`, `MockChainProvider`) for examples
