use async_trait::async_trait;

use crate::core::Message;

pub mod sliding;
pub mod summary;

pub use sliding::SlidingWindowMemory;
pub use summary::SummaryMemory;

/// Conversation memory strategy used by `GraphAgent`.
///
/// Abstracting it behind a trait lets the agent swap the bounded sliding window
/// for the LLM-summarising implementation without knowing either concrete type.
#[async_trait]
pub trait ConversationMemory: Send {
    async fn add(&mut self, message: Message);
    fn get_history(&self) -> Vec<Message>;
}
