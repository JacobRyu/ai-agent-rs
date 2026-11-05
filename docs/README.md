# ai-agent-rs Documentation

A local-first AI agent in Rust. Runs against Ollama, OpenAI, or Anthropic, uses a graph-based
ReAct orchestrator, ships 5 built-in tools plus an optional RAG tool, local retrieval via
candle + LanceDB, and sliding-window conversation memory.

## How These Docs Split

| Question | Answered by |
|----------|-------------|
| *What is it, and how does it work?* | [Architecture](architecture.md) — diagrams, module layout, control flow, data flow |
| *Why was it built this way?* | [Design Notes](design.md) — rationale, trade-offs, rejected alternatives (Japanese) |
| *How do I use it?* | [Getting Started](getting-started.md) |
| *How do I extend it?* | [Guides](guides/) |
| *What are the exact signatures?* | [API Reference](reference/api.md) |

Each fact is written down in exactly one place. Architecture owns the diagrams; Design Notes
owns the reasoning; the reference directory owns the exhaustive tables.

## Contents

### Start here

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Prerequisites, build, first run, indexing documents |
| [Architecture](architecture.md) | Mermaid diagrams for composition, dependencies, execution, and RAG, plus the per-module file map |

### Understand the design

| Document | Description |
|----------|-------------|
| [Design Notes](design.md) | Japanese-language rationale: why a graph orchestrator, why `Box<dyn Trait>`, what was rejected and why |

### Extend the agent

| Document | Description |
|----------|-------------|
| [Adding a Tool](guides/adding-a-tool.md) | Implement the `Tool` trait and register it |
| [Adding an LLM Provider](guides/adding-a-provider.md) | Implement `LLMProvider`, including streaming, and wire it into `main.rs` |
| [Adding a Graph Node](guides/adding-a-graph-node.md) | Implement `Node`, register it, and route to it |

### Look things up

| Document | Description |
|----------|-------------|
| [Configuration Reference](reference/configuration.md) | Every environment variable and its default — the single source of truth |
| [API Reference](reference/api.md) | Trait signatures, key structs, and error types |

## Quick Links

- **Source Layout**: `src/` with `core/`, `llm/`, `agent/`, `tools/`, `rag/`, `memory/`
- **Entry Point**: `src/main.rs`
- **Tests**: `cargo test` (79 tests, no external services required)
- **CI**: GitHub Actions — fmt, check, clippy, nextest (on `main` and `develop`)

## Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core ReAct + Ollama + Basic Tools | ✅ Complete |
| 2 | RAG Integration (LanceDB + candle) | ✅ Complete |
| 2 (roadmap) | Document ingestion pipeline (`index` subcommand) | ✅ Complete |
| 3 | Graph Orchestrator (LangGraph-style) | ✅ Complete |
| 3 (roadmap) | State.context utilization + iteration limit | ✅ Complete |
| Cleanup | Dead code removal (`ReActAgent`) | ✅ Complete |
| 2 (roadmap) | Streaming responses | ✅ Complete |
| 2 (roadmap) | Index deduplication (hash-based) | ✅ Complete |
| 3 (roadmap) | Parallel tool execution (futures::join_all) | ✅ Complete |
| 4 | Cloud providers (OpenAI, Anthropic) | ✅ Complete |
| 5 | Web UI + persistence | ❌ Planned |
| 3 (roadmap) | Streaming UI integration | ❌ Planned |
| 3 (roadmap) | Anthropic provider | ✅ Complete |
| — | Long-term memory (`SummaryMemory`) | ✅ Complete (`MEMORY_STRATEGY=summary`) |
