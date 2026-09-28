use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;

use crate::core::AgentError;
use crate::tools::Tool;

/// Maximum number of **bytes** of combined stdout+stderr returned to the LLM.
/// Kept well below typical context budgets to keep tool results from dominating
/// the prompt.
const MAX_OUTPUT_BYTES: usize = 10_000;

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return the output"
    }

    fn parameters(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn call(&self, args: Value) -> Result<String, AgentError> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::ToolError("Missing 'command' argument".into()))?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
            .map_err(|e| AgentError::ToolError(format!("Failed to execute command: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&format!("STDERR:\n{}", stderr));
        }
        if !output.status.success() && result.is_empty() {
            result = format!("Command exited with code: {:?}", output.status.code());
        }

        if result.len() > MAX_OUTPUT_BYTES {
            // `MAX_OUTPUT_BYTES` is a byte budget, so snap back to a character
            // boundary to avoid splitting a multi-byte character (which would panic).
            result.truncate(result.floor_char_boundary(MAX_OUTPUT_BYTES));
            result.push_str(" ... (truncated to 10000 bytes)");
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo hello"});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[tokio::test]
    async fn test_bash_pipe() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo 'hello world' | wc -w"});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "2");
    }

    #[tokio::test]
    async fn test_bash_empty_command() {
        let tool = BashTool;
        let args = serde_json::json!({"command": ""});
        let result = tool.call(args).await.unwrap();
        assert_eq!(result.trim(), "");
    }

    #[tokio::test]
    async fn test_bash_missing_command() {
        let tool = BashTool;
        let args = serde_json::json!({});
        let err = tool.call(args).await.unwrap_err();
        assert!(matches!(err, AgentError::ToolError(_)));
    }

    #[tokio::test]
    async fn test_bash_exit_code() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "exit 42"});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("42"));
    }

    #[tokio::test]
    async fn test_bash_stderr() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo output && echo error >&2"});
        let result = tool.call(args).await.unwrap();
        assert!(result.contains("output"));
        assert!(result.contains("error"));
    }

    #[tokio::test]
    async fn test_bash_truncates_ascii_output() {
        let tool = BashTool;
        // 20_000 ASCII bytes is comfortably past the 10_000 byte budget.
        let args = serde_json::json!({"command": "for i in $(seq 1 20000); do echo -n a; done"});
        let result = tool.call(args).await.unwrap();
        assert!(result.ends_with(" ... (truncated to 10000 bytes)"));
        assert_eq!(result.len(), " ... (truncated to 10000 bytes)".len() + MAX_OUTPUT_BYTES);
    }

    #[tokio::test]
    async fn test_bash_truncation_respects_char_boundaries() {
        let tool = BashTool;
        // U+3042 is 3 bytes (E3 81 82), so a 10_000-byte cut would land mid-character.
        // This panicked before truncation was made boundary-safe. Octal escapes are
        // used because POSIX `printf` has no `\u` support.
        let args = serde_json::json!({"command": "for i in $(seq 1 6000); do printf '\\343\\201\\202'; done"});
        let result = tool.call(args).await.unwrap();

        assert!(result.ends_with(" ... (truncated to 10000 bytes)"));
        // The visible payload is a whole number of 3-byte characters.
        let payload = result.trim_end_matches(" ... (truncated to 10000 bytes)");
        assert_eq!(payload.len(), 9999);
        assert!(payload.chars().all(|c| c == '\u{3042}'));
    }

    #[tokio::test]
    async fn test_bash_does_not_truncate_short_output() {
        let tool = BashTool;
        let args = serde_json::json!({"command": "echo short"});
        let result = tool.call(args).await.unwrap();
        assert!(!result.contains("truncated"));
    }
}
