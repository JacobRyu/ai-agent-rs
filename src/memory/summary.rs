use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::core::{LLMProvider, Message, Role};
use crate::memory::ConversationMemory;

/// LLM-based conversation memory: summarizes old messages once the window
/// fills, retaining key context as a synthetic System message.
///
/// Selected at startup with `MEMORY_STRATEGY=summary`; `SlidingWindowMemory` is
/// the default.
pub struct SummaryMemory {
    messages: VecDeque<Message>,
    max_messages: usize,
    summary_threshold: usize,
    provider: Arc<dyn LLMProvider>,
    summary: Option<String>,
}

impl SummaryMemory {
    pub fn new(
        max_messages: usize,
        summary_threshold: usize,
        provider: Arc<dyn LLMProvider>,
    ) -> Self {
        Self { messages: VecDeque::new(), max_messages, summary_threshold, provider, summary: None }
    }

    pub async fn add(&mut self, message: Message) {
        self.messages.push_back(message);
        self.enforce_limit().await;
    }

    pub fn get_history(&self) -> Vec<Message> {
        let mut result = Vec::new();

        if let Some(ref summary) = self.summary {
            result.push(Message {
                role: Role::System,
                content: format!("Previous conversation summary:\n{}", summary),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        result.extend(self.messages.iter().cloned());
        result
    }

    async fn enforce_limit(&mut self) {
        while self.messages.len() > self.max_messages {
            if self.messages.len() > self.summary_threshold {
                self.summarize_old_messages().await;
            } else {
                let has_system = self.messages.iter().any(|m| matches!(m.role, Role::System));
                if has_system
                    && self.messages.front().is_some_and(|m| !matches!(m.role, Role::System))
                {
                    self.messages.pop_front();
                } else if let Some(pos) =
                    self.messages.iter().position(|m| !matches!(m.role, Role::System))
                {
                    self.messages.remove(pos);
                } else {
                    break;
                }
            }
        }
    }

    async fn summarize_old_messages(&mut self) {
        // Summarize at least one message so `enforce_limit` always makes progress,
        // even when `max_messages` is 1 (or 0) and the window holds a single message.
        let split = (self.messages.len() / 2).max(1);
        let messages_to_summarize: Vec<Message> =
            self.messages.iter().take(split).cloned().collect();

        if messages_to_summarize.is_empty() {
            return;
        }

        let conversation_text = messages_to_summarize
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::System => "System",
                    Role::Tool => "Tool",
                };
                format!("{}: {}", role, m.content)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Summarize the following conversation concisely, preserving key information, decisions, and context:\n\n{}",
            conversation_text
        );

        let summary_messages = vec![Message {
            role: Role::User,
            content: prompt,
            tool_calls: None,
            tool_call_id: None,
        }];

        match self.provider.chat(&summary_messages, &[]).await {
            Ok(response) => {
                if let Some(new_summary) = response.content {
                    let combined = match &self.summary {
                        Some(existing) => format!("{}\n\n{}", existing, new_summary),
                        None => new_summary,
                    };
                    self.summary = Some(combined);

                    let total = self.messages.len();
                    for _ in 0..split {
                        self.messages.pop_front();
                    }

                    debug!("Summarized {} messages, keeping {}", split, total - split);
                }
            }
            Err(e) => {
                debug!("Failed to summarize messages: {}, falling back to eviction", e);
                let total = self.messages.len();
                for _ in 0..split {
                    self.messages.pop_front();
                }
                debug!("Evicted {} messages after summarization failure", split.min(total));
            }
        }
    }
}

#[async_trait]
impl ConversationMemory for SummaryMemory {
    async fn add(&mut self, message: Message) {
        SummaryMemory::add(self, message).await;
    }

    fn get_history(&self) -> Vec<Message> {
        SummaryMemory::get_history(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AgentError, LLMResponse, ToolDef};
    use async_trait::async_trait;

    struct MockSummaryProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl LLMProvider for MockSummaryProvider {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDef],
        ) -> Result<LLMResponse, AgentError> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(LLMResponse { content: Some("Summary of conversation".into()), tool_calls: None })
        }

        fn model_name(&self) -> &str {
            "mock_summary"
        }
    }

    fn msg(role: Role, content: &str) -> Message {
        Message { role, content: content.into(), tool_calls: None, tool_call_id: None }
    }

    #[tokio::test]
    async fn test_summary_memory_basic() {
        let provider =
            Arc::new(MockSummaryProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut mem = SummaryMemory::new(5, 3, provider);

        mem.add(msg(Role::User, "hello")).await;
        mem.add(msg(Role::Assistant, "hi")).await;

        let history = mem.get_history();
        assert_eq!(history.len(), 2);
        assert!(history.iter().all(|m| !matches!(m.role, Role::System)));
    }

    #[tokio::test]
    async fn test_summary_memory_triggers_summary() {
        let provider =
            Arc::new(MockSummaryProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut mem = SummaryMemory::new(6, 3, provider);

        for i in 0..5 {
            mem.add(msg(Role::User, &format!("question {}", i))).await;
            mem.add(msg(Role::Assistant, &format!("answer {}", i))).await;
        }

        let history = mem.get_history();
        assert!(!history.is_empty());
        assert_eq!(history[0].role, Role::System);
        assert!(history[0].content.contains("Summary"));
    }

    #[tokio::test]
    async fn test_zero_limits_still_make_progress() {
        let provider =
            Arc::new(MockSummaryProvider { call_count: std::sync::atomic::AtomicUsize::new(0) });
        let mut mem = SummaryMemory::new(0, 0, provider);

        mem.add(msg(Role::User, "only message")).await;

        let history = mem.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }
}
