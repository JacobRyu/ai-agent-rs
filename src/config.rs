pub struct Config {
    pub ollama_host: String,
    pub ollama_model: String,
    pub vector_store_uri: String,
    pub vector_table_name: String,
    pub openai_api_key: Option<String>,
    pub openai_model: String,
    pub openai_base_url: String,
    pub anthropic_api_key: Option<String>,
    pub anthropic_model: String,
    pub anthropic_base_url: String,
    pub anthropic_max_tokens: usize,
    pub llm_provider: String,
    pub memory_strategy: String,
    pub memory_max_messages: usize,
    pub memory_summary_threshold: usize,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            ollama_host: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string()),
            ollama_model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string()),
            vector_store_uri: std::env::var("VECTOR_STORE_URI")
                .unwrap_or_else(|_| "data/lancedb".to_string()),
            vector_table_name: std::env::var("VECTOR_TABLE_NAME")
                .unwrap_or_else(|_| "documents".to_string()),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            openai_model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string()),
            openai_base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            anthropic_model: std::env::var("ANTHROPIC_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-5-5".to_string()),
            anthropic_base_url: std::env::var("ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com".to_string()),
            anthropic_max_tokens: env_usize("ANTHROPIC_MAX_TOKENS", 4096),
            llm_provider: std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "ollama".to_string()),
            memory_strategy: std::env::var("MEMORY_STRATEGY")
                .unwrap_or_else(|_| "sliding".to_string()),
            memory_max_messages: env_usize("MEMORY_MAX_MESSAGES", 20),
            memory_summary_threshold: env_usize("MEMORY_SUMMARY_THRESHOLD", 12),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = Config::from_env();
        assert_eq!(config.ollama_host, "http://localhost:11434");
        assert_eq!(config.ollama_model, "qwen3:8b");
    }

    #[test]
    fn test_custom_host() {
        temp_env::with_var("OLLAMA_HOST", Some("http://custom:8080"), || {
            let config = Config::from_env();
            assert_eq!(config.ollama_host, "http://custom:8080");
        });
    }

    #[test]
    fn test_custom_model() {
        temp_env::with_var("OLLAMA_MODEL", Some("llama3:latest"), || {
            let config = Config::from_env();
            assert_eq!(config.ollama_model, "llama3:latest");
        });
    }

    #[test]
    fn test_memory_defaults() {
        temp_env::with_vars_unset(
            ["MEMORY_STRATEGY", "MEMORY_MAX_MESSAGES", "MEMORY_SUMMARY_THRESHOLD"],
            || {
                let config = Config::from_env();
                assert_eq!(config.memory_strategy, "sliding");
                assert_eq!(config.memory_max_messages, 20);
                assert_eq!(config.memory_summary_threshold, 12);
            },
        );
    }

    #[test]
    fn test_memory_custom_values() {
        temp_env::with_vars(
            [
                ("MEMORY_STRATEGY", Some("summary")),
                ("MEMORY_MAX_MESSAGES", Some("30")),
                ("MEMORY_SUMMARY_THRESHOLD", Some("10")),
            ],
            || {
                let config = Config::from_env();
                assert_eq!(config.memory_strategy, "summary");
                assert_eq!(config.memory_max_messages, 30);
                assert_eq!(config.memory_summary_threshold, 10);
            },
        );
    }

    #[test]
    fn test_invalid_memory_number_falls_back_to_default() {
        temp_env::with_var("MEMORY_MAX_MESSAGES", Some("not-a-number"), || {
            let config = Config::from_env();
            assert_eq!(config.memory_max_messages, 20);
        });
    }

    #[test]
    fn test_anthropic_defaults() {
        temp_env::with_vars_unset(
            ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL", "ANTHROPIC_MAX_TOKENS"],
            || {
                let config = Config::from_env();
                assert_eq!(config.anthropic_api_key, None);
                assert_eq!(config.anthropic_model, "claude-sonnet-5-5");
                assert_eq!(config.anthropic_base_url, "https://api.anthropic.com");
                assert_eq!(config.anthropic_max_tokens, 4096);
            },
        );
    }

    #[test]
    fn test_anthropic_custom_values() {
        temp_env::with_vars(
            [
                ("ANTHROPIC_API_KEY", Some("sk-ant-test")),
                ("ANTHROPIC_MODEL", Some("claude-opus-5-5")),
                ("ANTHROPIC_BASE_URL", Some("https://example.test")),
                ("ANTHROPIC_MAX_TOKENS", Some("2048")),
            ],
            || {
                let config = Config::from_env();
                assert_eq!(config.anthropic_api_key.as_deref(), Some("sk-ant-test"));
                assert_eq!(config.anthropic_model, "claude-opus-5-5");
                assert_eq!(config.anthropic_base_url, "https://example.test");
                assert_eq!(config.anthropic_max_tokens, 2048);
            },
        );
    }
}
