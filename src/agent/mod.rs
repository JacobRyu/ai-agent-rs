pub mod graph;

use self::graph::{
    GraphRunner, State,
    nodes::{LLMNode, ToolNode},
};
use crate::core::{AgentError, LLMProvider, Message, Role};
use crate::memory::ConversationMemory;
use crate::tools::ToolEngine;
use std::sync::Arc;

pub struct GraphAgent {
    runner: GraphRunner,
    memory: Box<dyn ConversationMemory>,
}

impl GraphAgent {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        tools: ToolEngine,
        system_prompt: String,
        memory: Box<dyn ConversationMemory>,
    ) -> Self {
        let mut runner = GraphRunner::new("llm");

        let llm_node = Arc::new(LLMNode::new(provider, tools.clone(), system_prompt));
        let tool_node = Arc::new(ToolNode::new(tools));

        runner.add_node("llm", llm_node);
        runner.add_node("tools", tool_node);

        Self { runner, memory }
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String, AgentError> {
        let mut state = State::new();
        state.messages.extend(self.memory.get_history());
        state.messages.push(Message {
            role: Role::User,
            content: user_input.to_string(),
            tool_calls: None,
            tool_call_id: None,
        });

        let final_state = self.runner.run(state).await?;

        let history_len = self.memory.get_history().len();
        let new_messages: Vec<Message> =
            final_state.messages.iter().skip(history_len).cloned().collect();
        for msg in new_messages {
            self.memory.add(msg).await;
        }

        let last_response =
            final_state.messages.last().map(|m| m.content.clone()).unwrap_or_default();

        Ok(last_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{LLMResponse, ToolCall, ToolCallFunction, ToolDef};
    use crate::memory::{SlidingWindowMemory, SummaryMemory};
    use crate::tools::builtin::BashTool;
    use async_trait::async_trait;

    struct MockProvider;

    fn sliding_memory() -> Box<dyn ConversationMemory> {
        Box::new(SlidingWindowMemory::new(20))
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            Ok(LLMResponse { content: Some("mock response".into()), tool_calls: None })
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    struct MockChainProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl LLMProvider for MockChainProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if count == 0 {
                Ok(LLMResponse {
                    content: Some(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".into(),
                        function: ToolCallFunction {
                            name: "bash".into(),
                            arguments: serde_json::json!({"command": "echo tool_call"}),
                        },
                    }]),
                })
            } else {
                Ok(LLMResponse { content: Some("final answer".into()), tool_calls: None })
            }
        }

        fn model_name(&self) -> &str {
            "mock_chain"
        }
    }

    #[tokio::test]
    async fn test_direct_response() {
        let provider = Arc::new(MockProvider);
        let tools = ToolEngine::new();
        let mut agent = GraphAgent::new(provider, tools, "prompt".into(), sliding_memory());

        let result = agent.run("hello").await.unwrap();
        assert_eq!(result, "mock response");
    }

    #[tokio::test]
    async fn test_tool_execution_then_response() {
        let provider =
            Arc::new(MockChainProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut tools = ToolEngine::new();
        tools.register(Arc::new(BashTool));
        let mut agent = GraphAgent::new(provider, tools, "prompt".into(), sliding_memory());

        let result = agent.run("run a command").await.unwrap();
        assert_eq!(result, "final answer");
    }

    #[tokio::test]
    async fn test_context_is_populated_during_run() {
        let provider =
            Arc::new(MockChainProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut tools = ToolEngine::new();
        tools.register(Arc::new(BashTool));
        let mut agent = GraphAgent::new(provider, tools, "prompt".into(), sliding_memory());

        agent.run("run a command").await.unwrap();

        let history = agent.memory.get_history();
        assert!(!history.is_empty(), "memory should have messages after run");
    }

    #[tokio::test]
    async fn test_max_iterations() {
        struct LoopProvider;
        #[async_trait]
        impl LLMProvider for LoopProvider {
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: &[ToolDef],
            ) -> Result<LLMResponse, AgentError> {
                Ok(LLMResponse {
                    content: Some(String::new()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".into(),
                        function: ToolCallFunction {
                            name: "bash".into(),
                            arguments: serde_json::json!({"command": "echo loop"}),
                        },
                    }]),
                })
            }
            fn model_name(&self) -> &str {
                "loop"
            }
        }

        let provider = Arc::new(LoopProvider);
        let mut tools = ToolEngine::new();
        tools.register(Arc::new(BashTool));
        let mut agent = GraphAgent::new(provider, tools, "prompt".into(), sliding_memory());

        let result = agent.run("loop").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Max iterations"));
    }

    #[tokio::test]
    async fn test_empty_tool_call_filtered() {
        struct EmptyNameProvider;
        #[async_trait]
        impl LLMProvider for EmptyNameProvider {
            async fn chat(
                &self,
                _messages: &[Message],
                _tools: &[ToolDef],
            ) -> Result<LLMResponse, AgentError> {
                Ok(LLMResponse {
                    content: Some("fallback response".into()),
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".into(),
                        function: ToolCallFunction {
                            name: String::new(),
                            arguments: serde_json::json!({}),
                        },
                    }]),
                })
            }
            fn model_name(&self) -> &str {
                "empty"
            }
        }

        let provider = Arc::new(EmptyNameProvider);
        let tools = ToolEngine::new();
        let mut agent = GraphAgent::new(provider, tools, "prompt".into(), sliding_memory());

        let result = agent.run("test").await.unwrap();
        assert_eq!(result, "fallback response");
    }

    #[tokio::test]
    async fn test_summary_memory_retains_summary_across_turns() {
        let provider = Arc::new(MockProvider);
        let memory = Box::new(SummaryMemory::new(4, 2, provider.clone()));
        let mut agent = GraphAgent::new(provider, ToolEngine::new(), "prompt".into(), memory);

        for i in 0..6 {
            agent.run(&format!("question {}", i)).await.unwrap();
        }

        let history = agent.memory.get_history();
        assert_eq!(history[0].role, Role::System);
        assert!(history[0].content.starts_with("Previous conversation summary:"));
    }
}
