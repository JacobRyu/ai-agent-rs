# API Reference

## Core Traits

### `LLMProvider`

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn chat(&self, messages: &[Message], tools: &[ToolDef]) -> Result<LLMResponse, AgentError>;

    async fn chat_stream<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDef],
    ) -> Result<BoxStream<'a, Result<StreamingChunk, AgentError>>, AgentError>;

    fn model_name(&self) -> &str;

    fn supports_streaming(&self) -> bool;
}

pub struct StreamingChunk {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub done: bool,
}
```

`LLMNode` calls `chat_stream()` when `supports_streaming()` returns `true`, otherwise
`chat()`. Both bundled providers (`OllamaProvider`, `OpenAIProvider`) return `true`.
`chat_stream` and `supports_streaming` have default implementations, so a new provider only
needs `chat` and `model_name` to be usable.

### `Tool`

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    async fn call(&self, args: Value) -> Result<String, AgentError>;
}
```

### `Node`

```rust
#[async_trait]
pub trait Node: Send + Sync {
    async fn process(&self, state: &mut State) -> Result<(), AgentError>;
}
```

### `EmbeddingProvider`

```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, AgentError>;
    fn dimension(&self) -> usize;
}
```

### `VectorStore`

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add(&self, id: &str, vector: Vec<f32>, metadata: Value) -> Result<(), AgentError>;
    async fn search(&self, vector: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>, AgentError>;
}
```

## Key Structs

### `GraphAgent`

```rust
impl GraphAgent {
    pub fn new(provider: Box<dyn LLMProvider>, tools: ToolEngine, system_prompt: String) -> Self;
    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError>;
}
```

### `GraphRunner`

```rust
impl GraphRunner {
    pub fn new(entry_point: &str) -> Self;
    pub fn add_node(&mut self, name: &str, node: Arc<dyn Node>);
    pub async fn run(&self, state: State) -> Result<State, AgentError>;
}
```

### `ToolEngine`

```rust
impl ToolEngine {
    pub fn new() -> Self;
    pub fn register(&mut self, tool: Arc<dyn Tool>);
    pub fn get_defs(&self) -> Vec<ToolDef>;
    pub async fn execute(&self, tool_call: &ToolCall) -> Result<ToolResult, AgentError>;
}
```

### `SlidingWindowMemory`

```rust
impl SlidingWindowMemory {
    pub fn new(max_messages: usize) -> Self;
    pub fn add(&mut self, message: Message);
    pub fn get_history(&self) -> Vec<Message>;
}
```

### `State`

```rust
pub struct State {
    pub messages: Vec<Message>,
    pub context: HashMap<String, Value>,
    pub next_node: Option<String>,
}
```

### `Indexer`

```rust
impl Indexer {
    pub fn new(embedding_provider: Arc<dyn EmbeddingProvider>, vector_store: Arc<dyn VectorStore>) -> Self;
    pub async fn index_directory(&mut self, dir_path: &str) -> Result<usize, AgentError>;
    pub fn skipped_duplicates(&self) -> usize;
    pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String>;
}
```

`index_directory` takes `&mut self` because it accumulates the set of seen content hashes
across a run. It returns the number of chunks indexed; call `skipped_duplicates()`
separately for the deduplication count.

`chunk_text` treats `chunk_size` as a **byte** budget and snaps to a character boundary, so
multi-byte text is never split mid-character. It prefers to break on a newline within the
last 50 bytes of each window.

## Error Types

`AgentError` variants:

| Variant | Source |
|---------|--------|
| `LLMError(String)` | LLM API failures |
| `ToolError(String)` | Tool execution errors |
| `SerdeError` | JSON serialization/deserialization |
| `IOError` | Filesystem I/O |
| `ToolNotFound(String)` | Missing tool in registry |
| `RAGError(String)` | RAG pipeline errors |
| `CandleError` | Candle ML framework errors |
| `VectorStoreError(String)` | LanceDB errors |
| `EmbeddingError(String)` | Embedding provider errors |
