mod agent;
mod config;
mod core;
mod llm;
mod memory;
mod rag;
mod tools;

use std::sync::Arc;

use tracing::{info, warn};

use crate::agent::GraphAgent;
use crate::config::Config;
use crate::core::{AgentError, LLMProvider};
use crate::llm::ollama::OllamaProvider;
use crate::llm::openai::OpenAIProvider;
use crate::memory::{ConversationMemory, SlidingWindowMemory, SummaryMemory};
use crate::rag::{
    CandleEmbeddingProvider, EmbeddingProvider, Indexer, LanceVectorStore, VectorStore,
};
use crate::tools::ToolEngine;
use crate::tools::builtin::*;

fn parse_args() -> Option<String> {
    let mut args = std::env::args().skip(1);
    let subcommand = args.next()?;
    if subcommand == "index" {
        args.next()
    } else {
        eprintln!("Usage: ai-agent-rs [index <path>]");
        std::process::exit(1);
    }
}

fn create_provider(config: &Config) -> Result<Arc<dyn LLMProvider>, AgentError> {
    match config.llm_provider.as_str() {
        "openai" => {
            let api_key = config.openai_api_key.as_ref().ok_or_else(|| {
                AgentError::LLMError(
                    "OPENAI_API_KEY must be set when LLM_PROVIDER=openai".to_string(),
                )
            })?;
            Ok(Arc::new(OpenAIProvider::with_base_url(
                api_key.clone(),
                config.openai_model.clone(),
                config.openai_base_url.clone(),
            )))
        }
        _ => Ok(Arc::new(OllamaProvider::new(
            config.ollama_host.clone(),
            config.ollama_model.clone(),
        ))),
    }
}

fn create_memory(config: &Config, provider: Arc<dyn LLMProvider>) -> Box<dyn ConversationMemory> {
    match config.memory_strategy.as_str() {
        "summary" => {
            info!(
                "Using SummaryMemory (max {} messages, summarize above {})",
                config.memory_max_messages, config.memory_summary_threshold
            );
            Box::new(SummaryMemory::new(
                config.memory_max_messages,
                config.memory_summary_threshold,
                provider,
            ))
        }
        "sliding" => Box::new(SlidingWindowMemory::new(config.memory_max_messages)),
        other => {
            warn!("Unknown MEMORY_STRATEGY '{}'; falling back to sliding", other);
            Box::new(SlidingWindowMemory::new(config.memory_max_messages))
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let config = Config::from_env();

    let embedding_provider: Option<Arc<dyn EmbeddingProvider>> =
        match CandleEmbeddingProvider::new() {
            Ok(p) => Some(Arc::new(p)),
            Err(e) => {
                warn!(
                    "Failed to initialize embedding provider: {}. RAG features will be disabled.",
                    e
                );
                None
            }
        };

    let vector_store: Option<Arc<dyn VectorStore>> = if let Some(ref ep) = embedding_provider {
        match LanceVectorStore::new(
            &config.vector_store_uri,
            &config.vector_table_name,
            ep.dimension(),
        )
        .await
        {
            Ok(vs) => Some(Arc::new(vs)),
            Err(e) => {
                warn!("Failed to initialize vector store: {}. RAG features will be disabled.", e);
                None
            }
        }
    } else {
        None
    };

    if let Some(path) = parse_args() {
        return run_index(path, embedding_provider, vector_store).await;
    }

    let provider = create_provider(&config)?;
    let memory = create_memory(&config, provider.clone());

    let mut tool_engine = ToolEngine::new();
    tool_engine.register(Arc::new(ReadFileTool));
    tool_engine.register(Arc::new(WriteFileTool));
    tool_engine.register(Arc::new(GlobTool));
    tool_engine.register(Arc::new(GrepTool));
    tool_engine.register(Arc::new(BashTool));

    if let (Some(ep), Some(vs)) = (embedding_provider, vector_store) {
        tool_engine.register(Arc::new(SearchDocsTool::new(ep, vs)));
        info!("RAG features enabled.");
    }

    let system_prompt = r#"You are an AI assistant that helps users by using tools.

Available tools (use these by their exact name):
- read_file(path: string): Read file contents
- write_file(path: string, content: string): Write content to a file
- glob(pattern: string): Search for files matching a glob pattern
- grep(pattern: string, path: string): Search file contents with regex
- bash(command: string): Execute shell commands
- search_docs(query: string, top_k: integer): Search for relevant information in the documentation

When you need to use a tool, the system will handle the function call for you.
Just reason about what tool to use and the system will execute it.
After receiving tool results, continue helping the user with the information.
If you can answer without tools, just respond directly."#
        .to_string();

    info!(
        "AI Agent started with provider: {}, model: {}",
        config.llm_provider,
        provider.model_name()
    );
    println!("AI Agent (Provider: {}, Model: {})", config.llm_provider, provider.model_name());

    let mut agent = GraphAgent::new(provider, tool_engine, system_prompt, memory);
    println!("Type 'exit' to quit");
    println!("{}", "-".repeat(40));

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        match agent.run(&input).await {
            Ok(response) => println!("{}", response),
            Err(e) => eprintln!("Error: {}", e),
        }
        println!();
    }

    Ok(())
}

async fn run_index(
    path: String,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    vector_store: Option<Arc<dyn VectorStore>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ep, vs) = match (embedding_provider, vector_store) {
        (Some(ep), Some(vs)) => (ep, vs),
        _ => {
            eprintln!("RAG is not available. Cannot index documents.");
            std::process::exit(1);
        }
    };

    println!("Indexing documents in {} ...", path);

    let mut indexer = Indexer::new(ep, vs);
    let count = indexer.index_directory(&path).await?;
    let skipped = indexer.skipped_duplicates();
    info!("Indexing complete. {} chunks indexed, {} duplicates skipped.", count, skipped);
    println!("Indexed {} chunks ({} duplicates skipped) from {}", count, skipped, path);
    Ok(())
}
