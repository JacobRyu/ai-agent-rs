use std::collections::VecDeque;

use async_trait::async_trait;

use crate::core::{Message, Role};
use crate::memory::ConversationMemory;

pub struct SlidingWindowMemory {
    messages: VecDeque<Message>,
    max_messages: usize,
}

impl SlidingWindowMemory {
    pub fn new(max_messages: usize) -> Self {
        Self { messages: VecDeque::new(), max_messages }
    }

    pub fn add(&mut self, message: Message) {
        self.messages.push_back(message);
        self.enforce_limit();
    }

    pub fn get_history(&self) -> Vec<Message> {
        self.messages.iter().cloned().collect()
    }

    fn enforce_limit(&mut self) {
        while self.messages.len() > self.max_messages {
            let has_system = self.messages.iter().any(|m| matches!(m.role, Role::System));
            if has_system && self.messages.front().is_some_and(|m| !matches!(m.role, Role::System))
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

#[async_trait]
impl ConversationMemory for SlidingWindowMemory {
    async fn add(&mut self, message: Message) {
        SlidingWindowMemory::add(self, message);
    }

    fn get_history(&self) -> Vec<Message> {
        SlidingWindowMemory::get_history(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, content: &str) -> Message {
        Message { role, content: content.into(), tool_calls: None, tool_call_id: None }
    }

    #[test]
    fn test_add_and_get_history() {
        let mut mem = SlidingWindowMemory::new(10);
        mem.add(msg(Role::User, "hello"));
        mem.add(msg(Role::Assistant, "hi"));

        let history = mem.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].content, "hello");
        assert_eq!(history[1].content, "hi");
    }

    #[test]
    fn test_enforce_limit_drops_oldest() {
        let mut mem = SlidingWindowMemory::new(3);
        mem.add(msg(Role::User, "q1"));
        mem.add(msg(Role::Assistant, "a1"));
        mem.add(msg(Role::User, "q2"));
        mem.add(msg(Role::Assistant, "a2"));

        assert_eq!(mem.get_history().len(), 3);
        assert_eq!(mem.get_history()[0].content, "a1");
    }

    #[test]
    fn test_preserve_system_message() {
        let mut mem = SlidingWindowMemory::new(3);
        mem.add(msg(Role::System, "system prompt"));
        mem.add(msg(Role::User, "q1"));
        mem.add(msg(Role::Assistant, "a1"));
        mem.add(msg(Role::User, "q2"));

        let history = mem.get_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, Role::System);
        assert_eq!(history[0].content, "system prompt");
    }

    #[test]
    fn test_empty_history() {
        let mem = SlidingWindowMemory::new(10);
        assert!(mem.get_history().is_empty());
    }

    #[test]
    fn test_exactly_at_limit() {
        let mut mem = SlidingWindowMemory::new(3);
        mem.add(msg(Role::System, "sys"));
        mem.add(msg(Role::User, "u1"));
        mem.add(msg(Role::Assistant, "a1"));
        assert_eq!(mem.get_history().len(), 3);
    }

    #[test]
    fn test_system_only_stays() {
        let mut mem = SlidingWindowMemory::new(1);
        mem.add(msg(Role::System, "sys"));
        mem.add(msg(Role::User, "u1"));
        let history = mem.get_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, Role::System);
    }
}
