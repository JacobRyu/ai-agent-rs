# Getting Started

## Prerequisites

- **Rust** (latest stable) — install via [rustup](https://rustup.rs/)
- **Ollama** — install from [ollama.com](https://ollama.com/), *or* an OpenAI or Anthropic API key
- A compatible model (Ollama default: `qwen3:8b`)

```bash
ollama pull qwen3:8b
```

## Build & Run

```bash
# Build
cargo build

# Run CLI
cargo run

# Run tests
cargo test
```

The agent starts an interactive REPL. Type your message at the `>` prompt. Enter `exit` or `quit` to stop.

## Configuration

Everything is configured with environment variables. The
[Configuration Reference](reference/configuration.md) is the single source of truth for the
full list, defaults, and per-variable detail — this page only shows what you need for a first
run.

| Variable | Default | Why you would set it |
|----------|---------|----------------------|
| `LLM_PROVIDER` | `ollama` | Switch to `openai` or `anthropic`. **This is what selects the backend** — setting an API key alone does nothing. |
| `OLLAMA_MODEL` | `qwen3:8b` | Use a different local model |
| `OPENAI_API_KEY` | (none) | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | (none) | Required when `LLM_PROVIDER=anthropic` |

```bash
# Default: local Ollama with the default model
cargo run

# Pick a different local model
OLLAMA_MODEL=llama3:latest cargo run

# Or use OpenAI instead
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... cargo run

# Or use Anthropic
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... cargo run
```

## What Happens on Startup

1. `Config::from_env()` reads environment variables
2. The provider is built from `LLM_PROVIDER` (`OllamaProvider`, `OpenAIProvider`, or `AnthropicProvider`)
3. `CandleEmbeddingProvider` loads a BERT model (`all-MiniLM-L6-v2`) from HuggingFace
4. `LanceVectorStore` opens or creates the vector database
5. `ToolEngine` registers 5 built-in filesystem/shell tools
6. `GraphAgent` wires LLM + tools + memory together
7. The REPL loop begins

If RAG initialization fails (no HuggingFace access, missing model), the agent **still starts**
with all other features — `SearchDocsTool` is simply not registered, so only `search_docs` is
unavailable. With RAG enabled, 6 tools are registered. The only startups that abort are
selecting a cloud provider without its API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`),
which exits with an `AgentError::LLMError`.

For the full call-by-call startup sequence, see
[Architecture §4「Startup Sequence」](architecture.md#4-startup-sequence).

## Indexing Documents for RAG

The vector store is **empty** by default. To populate it, use the `index` subcommand:

```bash
cargo run -- index ./docs
```

This walks the directory tree and, for each file with a `.md`, `.txt`, `.rs`, `.toml`,
`.json`, `.yaml`, or `.yml` extension:

- splits the content into chunks of at most **512 bytes**, preferring to break on a
  newline near the end of each window
- skips any chunk whose content hash has already been indexed in this run
  (re-indexing the same tree does not create duplicate entries)
- embeds each chunk with BERT and stores the vector in LanceDB

The command reports the number of chunks indexed, files processed, and duplicates skipped.
Any single file that fails to read is logged and skipped rather than aborting the run.

To see how this connects to the `search_docs` tool, see
[Architecture §6「RAG Pipelines」](architecture.md#6-rag-pipelines).
