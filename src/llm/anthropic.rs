use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use tracing::debug;

use crate::core::provider::StreamingChunk;
use crate::core::{
    AgentError, LLMProvider, LLMResponse, Message, Role, ToolCall, ToolCallFunction, ToolDef,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn with_base_url(
        api_key: String,
        model: String,
        base_url: String,
        max_tokens: u32,
    ) -> Self {
        Self { client: Client::new(), api_key, model, base_url, max_tokens }
    }

    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        stream: bool,
    ) -> AnthropicRequest {
        let (system, messages) = convert_messages(messages);

        AnthropicRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            messages,
            system,
            tools: tools.iter().cloned().map(AnthropicTool::from).collect(),
            stream: stream.then_some(true),
        }
    }
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize, Debug)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

impl From<ToolDef> for AnthropicTool {
    fn from(tool: ToolDef) -> Self {
        AnthropicTool {
            name: tool.name,
            description: tool.description,
            input_schema: tool.parameters,
        }
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseBlock>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<AnthropicMessage>) {
    let mut system_parts = Vec::new();
    let mut converted = Vec::new();

    for message in messages {
        match message.role {
            Role::System => system_parts.push(message.content.clone()),
            Role::User => push_content(
                &mut converted,
                "user",
                AnthropicContent::Text { text: message.content.clone() },
            ),
            Role::Assistant => {
                let mut has_block = false;

                if !message.content.is_empty() {
                    push_content(
                        &mut converted,
                        "assistant",
                        AnthropicContent::Text { text: message.content.clone() },
                    );
                    has_block = true;
                }

                if let Some(tool_calls) = &message.tool_calls {
                    for call in tool_calls {
                        push_content(
                            &mut converted,
                            "assistant",
                            AnthropicContent::ToolUse {
                                id: call.id.clone(),
                                name: call.function.name.clone(),
                                input: call.function.arguments.clone(),
                            },
                        );
                        has_block = true;
                    }
                }

                if !has_block {
                    debug!("Skipping empty assistant message");
                }
            }
            Role::Tool => push_tool_result(
                &mut converted,
                AnthropicContent::ToolResult {
                    tool_use_id: message.tool_call_id.clone().unwrap_or_default(),
                    content: message.content.clone(),
                },
            ),
        }
    }

    let system = if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) };

    (system, converted)
}

fn push_content(messages: &mut Vec<AnthropicMessage>, role: &str, content: AnthropicContent) {
    if let Some(last) = messages.last_mut()
        && last.role == role
    {
        last.content.push(content);
        return;
    }

    messages.push(AnthropicMessage { role: role.to_string(), content: vec![content] });
}

fn push_tool_result(messages: &mut Vec<AnthropicMessage>, content: AnthropicContent) {
    if let Some(last) = messages.last_mut()
        && last.role == "user"
        && last.content.iter().all(|block| matches!(block, AnthropicContent::ToolResult { .. }))
    {
        last.content.push(content);
        return;
    }

    messages.push(AnthropicMessage { role: "user".to_string(), content: vec![content] });
}

fn convert_response(response: AnthropicResponse) -> LLMResponse {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in response.content {
        match block {
            AnthropicResponseBlock::Text { text: part } => text.push_str(&part),
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                tool_calls
                    .push(ToolCall { id, function: ToolCallFunction { name, arguments: input } });
            }
            AnthropicResponseBlock::Other => {}
        }
    }

    LLMResponse {
        content: if text.is_empty() { None } else { Some(text) },
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
    }
}

#[derive(Default)]
struct StreamToolUse {
    id: String,
    name: String,
    json: String,
}

fn handle_stream_event(
    event: &Value,
    texts: &mut BTreeMap<usize, String>,
    tools: &mut BTreeMap<usize, StreamToolUse>,
) -> Result<(), AgentError> {
    let event_type = event.get("type").and_then(Value::as_str).unwrap_or_default();
    let index = event.get("index").and_then(Value::as_u64).map(|value| value as usize);

    match event_type {
        "content_block_start" => {
            let Some(index) = index else { return Ok(()) };
            let Some(block) = event.get("content_block") else { return Ok(()) };

            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let text = block.get("text").and_then(Value::as_str).unwrap_or_default();
                    texts.insert(index, text.to_string());
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                    let name = block.get("name").and_then(Value::as_str).unwrap_or_default();
                    let initial = match block.get("input") {
                        Some(Value::Object(map)) if !map.is_empty() => {
                            serde_json::to_string(&Value::Object(map.clone())).unwrap_or_default()
                        }
                        _ => String::new(),
                    };

                    tools.insert(
                        index,
                        StreamToolUse { id: id.to_string(), name: name.to_string(), json: initial },
                    );
                }
                _ => {}
            }
        }
        "content_block_delta" => {
            let Some(index) = index else { return Ok(()) };
            let Some(delta) = event.get("delta") else { return Ok(()) };

            match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => {
                    let text = delta.get("text").and_then(Value::as_str).unwrap_or_default();
                    texts.entry(index).or_default().push_str(text);
                }
                Some("input_json_delta") => {
                    let partial =
                        delta.get("partial_json").and_then(Value::as_str).unwrap_or_default();
                    tools.entry(index).or_default().json.push_str(partial);
                }
                _ => {}
            }
        }
        "error" => {
            let message = event
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown streaming error");
            return Err(AgentError::LLMError(format!("API stream error: {}", message)));
        }
        _ => {}
    }

    Ok(())
}

fn process_stream_line(
    line: &str,
    texts: &mut BTreeMap<usize, String>,
    tools: &mut BTreeMap<usize, StreamToolUse>,
) -> Result<bool, AgentError> {
    let Some(data) = line.strip_prefix("data:") else { return Ok(false) };
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(false);
    }

    match serde_json::from_str::<Value>(data) {
        Ok(event) => {
            handle_stream_event(&event, texts, tools)?;
            Ok(event.get("type").and_then(Value::as_str) == Some("message_stop"))
        }
        Err(error) => {
            debug!("Failed to parse Anthropic streaming line: {} - line: {}", error, line);
            Ok(false)
        }
    }
}

fn finalize_stream(
    texts: &BTreeMap<usize, String>,
    tools: &BTreeMap<usize, StreamToolUse>,
) -> StreamingChunk {
    let content: String = texts.values().cloned().collect();

    let tool_calls: Vec<ToolCall> = tools
        .values()
        .map(|tool| {
            let arguments = if tool.json.trim().is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&tool.json)
                    .unwrap_or_else(|_| Value::String(tool.json.clone()))
            };

            ToolCall {
                id: tool.id.clone(),
                function: ToolCallFunction { name: tool.name.clone(), arguments },
            }
        })
        .collect();

    StreamingChunk {
        content: Some(content),
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        done: true,
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<LLMResponse, AgentError> {
        let url = format!("{}/v1/messages", self.base_url);
        let request = self.build_request(messages, tools, false);

        debug!(
            "Anthropic request: model={}, messages={}, tools={}",
            self.model,
            request.messages.len(),
            request.tools.len()
        );

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::LLMError(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AgentError::LLMError(format!("API error ({}): {}", status, body)));
        }

        let resp_text = resp
            .text()
            .await
            .map_err(|e| AgentError::LLMError(format!("Read body failed: {}", e)))?;

        debug!("Raw Anthropic response: {}", resp_text);

        let response: AnthropicResponse = serde_json::from_str(&resp_text).map_err(|e| {
            AgentError::LLMError(format!(
                "Parse failed: {} - body: {}",
                e,
                &resp_text[..resp_text.len().min(500)]
            ))
        })?;

        Ok(convert_response(response))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_stream<'a>(
        &'a self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<BoxStream<'a, Result<StreamingChunk, AgentError>>, AgentError> {
        let url = format!("{}/v1/messages", self.base_url);
        let request = self.build_request(messages, tools, true);

        debug!(
            "Anthropic streaming request: model={}, messages={}, tools={}",
            self.model,
            request.messages.len(),
            request.tools.len()
        );

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::LLMError(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AgentError::LLMError(format!("API error ({}): {}", status, body)));
        }

        let byte_stream = resp.bytes_stream();

        let stream = futures::stream::unfold(
            (
                byte_stream,
                String::new(),
                BTreeMap::<usize, String>::new(),
                BTreeMap::<usize, StreamToolUse>::new(),
                false,
            ),
            |(mut byte_stream, mut buffer, mut texts, mut tools, done)| async move {
                if done {
                    return None;
                }

                'stream: loop {
                    while let Some(position) = buffer.find('\n') {
                        let line = buffer[..position].trim_end_matches('\r').to_string();
                        buffer.drain(..=position);

                        match process_stream_line(&line, &mut texts, &mut tools) {
                            Ok(true) => break 'stream,
                            Ok(false) => {}
                            Err(error) => {
                                return Some((
                                    Err(error),
                                    (byte_stream, buffer, texts, tools, true),
                                ));
                            }
                        }
                    }

                    match byte_stream.next().await {
                        Some(Ok(chunk)) => {
                            buffer.push_str(&String::from_utf8_lossy(&chunk));
                        }
                        Some(Err(error)) => {
                            return Some((
                                Err(AgentError::LLMError(format!("Stream error: {}", error))),
                                (byte_stream, buffer, texts, tools, true),
                            ));
                        }
                        None => break,
                    }
                }

                if !buffer.is_empty() {
                    let line = buffer.trim_end_matches('\r').to_string();
                    buffer.clear();
                    if let Err(error) = process_stream_line(&line, &mut texts, &mut tools) {
                        return Some((Err(error), (byte_stream, buffer, texts, tools, true)));
                    }
                }

                let chunk = finalize_stream(&texts, &tools);
                Some((
                    Ok(chunk),
                    (byte_stream, String::new(), BTreeMap::new(), BTreeMap::new(), true),
                ))
            },
        );

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn message(role: Role, content: &str) -> Message {
        Message { role, content: content.to_string(), tool_calls: None, tool_call_id: None }
    }

    #[test]
    fn test_system_messages_become_top_level_system() {
        let messages = vec![
            message(Role::System, "be helpful"),
            message(Role::User, "hello"),
            message(Role::System, "be brief"),
        ];

        let (system, converted) = convert_messages(&messages);

        assert_eq!(system.as_deref(), Some("be helpful\n\nbe brief"));
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");

        let value = serde_json::to_value(&converted[0]).unwrap();
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][0]["text"], "hello");
    }

    #[test]
    fn test_assistant_tool_calls_become_tool_use_blocks() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: "Let me check".to_string(),
            tool_calls: Some(vec![ToolCall {
                id: "toolu_1".to_string(),
                function: ToolCallFunction {
                    name: "glob".to_string(),
                    arguments: json!({"pattern": "*.rs"}),
                },
            }]),
            tool_call_id: None,
        }];

        let (_, converted) = convert_messages(&messages);
        let value = serde_json::to_value(&converted[0]).unwrap();

        assert_eq!(value["role"], "assistant");
        assert_eq!(value["content"][0]["type"], "text");
        assert_eq!(value["content"][1]["type"], "tool_use");
        assert_eq!(value["content"][1]["id"], "toolu_1");
        assert_eq!(value["content"][1]["name"], "glob");
        assert_eq!(value["content"][1]["input"]["pattern"], "*.rs");
    }

    #[test]
    fn test_consecutive_tool_results_share_one_user_message() {
        let messages = vec![
            Message {
                role: Role::Tool,
                content: "first".to_string(),
                tool_calls: None,
                tool_call_id: Some("toolu_1".to_string()),
            },
            Message {
                role: Role::Tool,
                content: "second".to_string(),
                tool_calls: None,
                tool_call_id: Some("toolu_2".to_string()),
            },
        ];

        let (_, converted) = convert_messages(&messages);

        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content.len(), 2);

        let value = serde_json::to_value(&converted[0]).unwrap();
        assert_eq!(value["content"][0]["type"], "tool_result");
        assert_eq!(value["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(value["content"][0]["content"], "first");
        assert_eq!(value["content"][1]["tool_use_id"], "toolu_2");
    }

    #[test]
    fn test_tool_results_are_not_merged_into_user_text() {
        let messages = vec![
            message(Role::User, "hello"),
            Message {
                role: Role::Tool,
                content: "result".to_string(),
                tool_calls: None,
                tool_call_id: Some("toolu_1".to_string()),
            },
        ];

        let (_, converted) = convert_messages(&messages);

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[1].role, "user");
        assert!(matches!(converted[0].content[0], AnthropicContent::Text { .. }));
        assert!(matches!(converted[1].content[0], AnthropicContent::ToolResult { .. }));
    }

    #[test]
    fn test_tool_definitions_use_input_schema() {
        let tools = vec![ToolDef {
            name: "glob".to_string(),
            description: "Find files".to_string(),
            parameters: json!({"type": "object", "properties": {"pattern": {"type": "string"}}}),
        }];

        let converted: Vec<AnthropicTool> = tools.into_iter().map(AnthropicTool::from).collect();
        let value = serde_json::to_value(&converted[0]).unwrap();

        assert_eq!(value["name"], "glob");
        assert_eq!(value["description"], "Find files");
        assert_eq!(value["input_schema"]["type"], "object");
        assert!(value.get("parameters").is_none());
    }

    #[test]
    fn test_request_serialization_includes_stream_and_max_tokens() {
        let provider = AnthropicProvider::with_base_url(
            "test-key".to_string(),
            "claude-sonnet-5-5".to_string(),
            "https://api.anthropic.com".to_string(),
            1024,
        );
        let messages = vec![message(Role::System, "be helpful"), message(Role::User, "hi")];
        let tools = vec![ToolDef {
            name: "glob".to_string(),
            description: "Find files".to_string(),
            parameters: json!({"type": "object"}),
        }];

        let request = provider.build_request(&messages, &tools, true);
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "claude-sonnet-5-5");
        assert_eq!(value["max_tokens"], 1024);
        assert_eq!(value["stream"], true);
        assert_eq!(value["system"], "be helpful");
        assert_eq!(value["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn test_response_parsing_handles_text_and_tool_use() {
        let response: AnthropicResponse = serde_json::from_value(json!({
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
                {"type": "tool_use", "id": "toolu_1", "name": "glob", "input": {"pattern": "*.rs"}},
                {"type": "thinking", "thinking": "internal"}
            ]
        }))
        .unwrap();

        let converted = convert_response(response);

        assert_eq!(converted.content.as_deref(), Some("Hello world"));
        let tool_calls = converted.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "toolu_1");
        assert_eq!(tool_calls[0].function.name, "glob");
        assert_eq!(tool_calls[0].function.arguments["pattern"], "*.rs");
    }

    #[test]
    fn test_stream_events_accumulate_text_and_tool_input() {
        let mut texts = BTreeMap::new();
        let mut tools = BTreeMap::new();

        let events = [
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi "}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "there"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "glob", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"pattern\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"*.rs\"}"}}),
            json!({"type": "message_stop"}),
        ];

        for event in events {
            handle_stream_event(&event, &mut texts, &mut tools).unwrap();
        }

        let chunk = finalize_stream(&texts, &tools);

        assert_eq!(chunk.content.as_deref(), Some("Hi there"));
        let tool_calls = chunk.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "glob");
        assert_eq!(tool_calls[0].function.arguments["pattern"], "*.rs");
        assert!(chunk.done);
    }

    #[test]
    fn test_process_stream_line_detects_message_stop() {
        let mut texts = BTreeMap::new();
        let mut tools = BTreeMap::new();

        let line = r#"data: {"type":"message_stop"}"#;

        assert!(process_stream_line(line, &mut texts, &mut tools).unwrap());
    }

    #[test]
    fn test_stream_error_event_is_reported() {
        let mut texts = BTreeMap::new();
        let mut tools = BTreeMap::new();

        let event = json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}
        });

        let error = handle_stream_event(&event, &mut texts, &mut tools).unwrap_err();
        assert!(error.to_string().contains("Overloaded"));
    }
}
