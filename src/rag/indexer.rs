use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use tokio::fs;
use tracing::{debug, info, warn};

use crate::core::AgentError;
use crate::rag::{EmbeddingProvider, VectorStore};

pub struct Indexer {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_store: Arc<dyn VectorStore>,
    chunk_size: usize,
    seen_hashes: HashSet<String>,
    skipped_chunks: usize,
}

impl Indexer {
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_store: Arc<dyn VectorStore>,
    ) -> Self {
        Self {
            embedding_provider,
            vector_store,
            chunk_size: 512,
            seen_hashes: HashSet::new(),
            skipped_chunks: 0,
        }
    }

    pub fn skipped_duplicates(&self) -> usize {
        self.skipped_chunks
    }

    fn content_hash(text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub async fn index_directory(&mut self, dir_path: &str) -> Result<usize, AgentError> {
        let root = PathBuf::from(dir_path);
        if !root.is_dir() {
            return Err(AgentError::RAGError(format!("Not a directory: {}", dir_path)));
        }

        let mut total_chunks = 0usize;
        let mut total_files = 0usize;
        let mut error_count = 0usize;
        let mut pending = VecDeque::new();
        pending.push_back(root);

        while let Some(dir) = pending.pop_front() {
            let mut entries = match fs::read_dir(&dir).await {
                Ok(e) => e,
                Err(e) => {
                    warn!("Skipping directory '{}': {}", dir.display(), e);
                    error_count += 1;
                    continue;
                }
            };

            while let Some(entry) = entries.next_entry().await? {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    pending.push_back(entry_path);
                    continue;
                }

                let ext =
                    entry_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if !matches!(ext.as_str(), "md" | "txt" | "rs" | "toml" | "json" | "yaml" | "yml") {
                    continue;
                }

                let content = match fs::read_to_string(&entry_path).await {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("Skipping '{}': {}", entry_path.display(), e);
                        error_count += 1;
                        continue;
                    }
                };

                let chunks = Self::chunk_text(&content, self.chunk_size);
                let file_name = entry_path.to_string_lossy().to_string();
                let mut file_chunks = 0usize;

                for (i, chunk) in chunks.iter().enumerate() {
                    let hash = Self::content_hash(chunk);
                    if self.seen_hashes.contains(&hash) {
                        self.skipped_chunks += 1;
                        debug!("Skipping duplicate chunk from '{}' index {}", file_name, i);
                        continue;
                    }
                    self.seen_hashes.insert(hash);

                    let doc_id = format!("{}#chunk{}", file_name, i);
                    match self.embedding_provider.embed(chunk).await {
                        Ok(vector) => {
                            let metadata = serde_json::json!({
                                "source": file_name,
                                "chunk_index": i,
                                "text": chunk,
                            });
                            match self.vector_store.add(&doc_id, vector, metadata).await {
                                Ok(()) => {
                                    total_chunks += 1;
                                    file_chunks += 1;
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to store chunk {} from '{}': {}",
                                        i, file_name, e
                                    );
                                    error_count += 1;
                                }
                            }
                        }
                        Err(e) => {
                            warn!("Failed to embed chunk {} from '{}': {}", i, file_name, e);
                            error_count += 1;
                        }
                    }
                }

                total_files += 1;
                info!(
                    "Indexed {} — {} chunks ({} total, {} errors)",
                    file_name, file_chunks, total_chunks, error_count
                );
            }
        }

        if error_count > 0 {
            warn!(
                "Indexing completed with {} errors. {} chunks indexed from {} files.",
                error_count, total_chunks, total_files
            );
        }

        Ok(total_chunks)
    }

    pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
        const BREAK_SEARCH_BYTES: usize = 50;

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let len = text.len();

        while start < len {
            // Preferred break point: the last newline within the trailing
            // `BREAK_SEARCH_BYTES` of the window, so paragraphs stay intact.
            let mut end = (start + chunk_size).min(len);
            if end < len {
                let search_start = end.saturating_sub(BREAK_SEARCH_BYTES);
                if let Some(offset) =
                    text.as_bytes()[search_start..end].iter().rposition(|&b| b == b'\n')
                {
                    end = search_start + offset + 1;
                }
            }

            // `chunk_size` is a byte budget, so snap back to a character boundary.
            // Without this, slicing a multi-byte document (e.g. Japanese) at a
            // non-boundary offset panics.
            end = text.floor_char_boundary(end);
            if end <= start {
                // Window landed inside a single character: emit that whole character.
                end = start + text[start..].chars().next().map_or(0, |c| c.len_utf8());
            }

            chunks.push(text[start..end].to_string());
            start = end;
        }

        if chunks.is_empty() && !text.is_empty() {
            chunks.push(text.to_string());
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_small() {
        let chunks = Indexer::chunk_text("hello world", 512);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn test_chunk_text_splits_at_newline() {
        let text = "aaaaa\nbbbbb\nccccc\nddddd\neeeee\nfffff\n";
        let chunks = Indexer::chunk_text(text, 20);
        assert!(chunks.len() >= 2, "should split into multiple chunks: {:?}", chunks);
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = Indexer::chunk_text("", 512);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = Indexer::content_hash("hello world");
        let h2 = Indexer::content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_different() {
        let h1 = Indexer::content_hash("hello");
        let h2 = Indexer::content_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chunk_text_multibyte_does_not_panic() {
        // 3 bytes per character, so a 512-byte window ends mid-character.
        // This panicked before chunk boundaries were made UTF-8 safe.
        let text = "あ".repeat(2000);
        let chunks = Indexer::chunk_text(&text, 512);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.is_empty()));
        assert!(chunks.iter().all(|c| c.chars().all(|ch| ch == 'あ')));
    }

    #[test]
    fn test_chunk_text_multibyte_round_trips() {
        let text = "これは日本語のドキュメントです。\n".repeat(500);
        let chunks = Indexer::chunk_text(&text, 512);
        assert_eq!(chunks.concat(), text, "chunking must not lose or corrupt text");
    }

    #[test]
    fn test_chunk_text_emoji_and_ascii_mixed() {
        // 4-byte characters mixed with ASCII: no cut may land mid-character.
        let text = "abc 🎉 def 🎉 ghi 🎉 ".repeat(400);
        let chunks = Indexer::chunk_text(&text, 300);
        assert_eq!(chunks.concat(), text);
        assert!(chunks.iter().all(|c| c.len() <= 300 || c.chars().count() == 1));
    }

    #[test]
    fn test_chunk_text_respects_budget() {
        // Every chunk fits the byte budget, except one that holds a single
        // character wider than the budget itself.
        let text = "a".repeat(5000);
        let chunks = Indexer::chunk_text(&text, 512);
        assert!(chunks.len() >= 9);
        assert!(chunks.iter().all(|c| c.len() <= 512));
    }
}
