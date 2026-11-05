# Configuration Reference

All configuration is through environment variables.

## Provider Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | Which backend to use: `ollama`, `openai`, or `anthropic`. Any other value falls back to `ollama`. |

`LLM_PROVIDER` is the only way to select a cloud backend. Setting `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` alone does not switch providers.

```bash
# Use Ollama (default)
LLM_PROVIDER=ollama cargo run

# Use OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... cargo run

# Use Anthropic
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... cargo run
```

If the selected provider's API key is unset, the agent exits at startup with an
`AgentError::LLMError` (no API calls are made). The messages are
`OPENAI_API_KEY must be set when LLM_PROVIDER=openai` and
`ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic`.

## Ollama

Used when `LLM_PROVIDER=ollama`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Model name for chat completions |

## OpenAI

Used when `LLM_PROVIDER=openai`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none, **required**) | API key. Read from the environment only; never passed on the command line. |
| `OPENAI_MODEL` | `gpt-4o` | Model name for chat completions |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL. Point this at any OpenAI-compatible endpoint (e.g. a local gateway). |

## Anthropic

Used when `LLM_PROVIDER=anthropic`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (none, **required**) | API key. Read from the environment only; never passed on the command line. |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model name for the Messages API |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | API base URL. The client appends `/v1/messages`. |
| `ANTHROPIC_MAX_TOKENS` | `4096` | Maximum output tokens per response. Required by the Messages API. |

All bundled providers support streaming responses, so no extra configuration is needed for
that.

## Conversation Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_STRATEGY` | `sliding` | `sliding` keeps the last N messages; `summary` summarises older messages with the LLM. Unknown values fall back to `sliding` with a warning. |
| `MEMORY_MAX_MESSAGES` | `20` | Maximum messages kept before eviction (or summarisation). |
| `MEMORY_SUMMARY_THRESHOLD` | `12` | Message count above which `summary` starts summarising. If it is greater than or equal to `MEMORY_MAX_MESSAGES`, eviction runs instead. |

`summary` issues one extra LLM request when the window overflows, using the same provider
as inference. If that request fails, it falls back to plain eviction. Invalid numeric
values fall back to the defaults.

```bash
MEMORY_STRATEGY=summary MEMORY_MAX_MESSAGES=30 MEMORY_SUMMARY_THRESHOLD=10 cargo run
```

## Vector Store

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_URI` | `data/lancedb` | Filesystem path for LanceDB storage |
| `VECTOR_TABLE_NAME` | `documents` | LanceDB table name for document vectors |

## General

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | (none) | Tracing/log level: `info`, `debug`, `warn`, `error` |

## Example

```bash
RUST_LOG=debug \
  LLM_PROVIDER=ollama \
  OLLAMA_HOST=http://192.168.1.100:11434 \
  OLLAMA_MODEL=llama3:70b \
  VECTOR_STORE_URI=/data/agent/lancedb \
  VECTOR_TABLE_NAME=my_docs \
  MEMORY_STRATEGY=summary \
  MEMORY_MAX_MESSAGES=30 \
  MEMORY_SUMMARY_THRESHOLD=10 \
  cargo run
```

```bash
RUST_LOG=info \
  LLM_PROVIDER=openai \
  OPENAI_API_KEY=sk-... \
  OPENAI_MODEL=gpt-4o \
  OPENAI_BASE_URL=https://api.openai.com/v1 \
  cargo run
```

```bash
RUST_LOG=info \
  LLM_PROVIDER=anthropic \
  ANTHROPIC_API_KEY=sk-ant-... \
  ANTHROPIC_MODEL=claude-sonnet-5-5 \
  ANTHROPIC_MAX_TOKENS=4096 \
  cargo run
```
