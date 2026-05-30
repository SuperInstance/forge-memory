//! # forge-memory
//!
//! External tile memory store for Plato agents.
//! Stores tiles with content hashes for retrieval. Uses sled for embedded storage
//! with zero network dependencies. Tiles serialized with bincode.

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sled::Db;
use std::path::Path;
use uuid::Uuid;

/// Kinds of tiles a Plato agent might store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TileKind {
    Text,
    Code,
    Audio,
    Image,
    Structured,
    Unknown(String),
}

/// A single tile of memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Tile {
    pub id: Uuid,
    pub kind: TileKind,
    pub content: Vec<u8>,
    pub source: Option<Uuid>,
    pub metadata: Vec<(String, String)>,
    pub embedding: Vec<f32>,
    pub created_at: u64,
}

/// Statistics about the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub total_tiles: u64,
    pub total_bytes: u64,
    pub kind_counts: Vec<(String, u64)>,
}

/// Key tree names.
const TILES_TREE: &str = "tiles";
const KIND_INDEX: &str = "kind_index";
const SOURCE_INDEX: &str = "source_index";

/// The tile memory store backed by sled.
pub struct TileStore {
    db: Db,
}

impl TileStore {
    /// Open or create a tile store at the given path.
    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    fn tiles_tree(&self) -> Result<sled::Tree, Box<dyn std::error::Error>> {
        Ok(self.db.open_tree(TILES_TREE)?)
    }

    fn kind_tree(&self) -> Result<sled::Tree, Box<dyn std::error::Error>> {
        Ok(self.db.open_tree(KIND_INDEX)?)
    }

    fn source_tree(&self) -> Result<sled::Tree, Box<dyn std::error::Error>> {
        Ok(self.db.open_tree(SOURCE_INDEX)?)
    }

    fn kind_key(kind: &TileKind, id: &Uuid) -> Vec<u8> {
        let kind_str = match kind {
            TileKind::Text => "Text",
            TileKind::Code => "Code",
            TileKind::Audio => "Audio",
            TileKind::Image => "Image",
            TileKind::Structured => "Structured",
            TileKind::Unknown(s) => s,
        };
        let mut key = format!("{}:", kind_str).into_bytes();
        key.extend_from_slice(id.as_bytes());
        key
    }

    fn source_key(source: &Uuid, id: &Uuid) -> Vec<u8> {
        let mut key = source.as_bytes().to_vec();
        key.extend_from_slice(id.as_bytes());
        key
    }

    /// Store tiles, returning their IDs.
    pub fn store(&self, tiles: &[Tile]) -> Result<Vec<Uuid>, Box<dyn std::error::Error>> {
        let tiles_tree = self.tiles_tree()?;
        let kind_tree = self.kind_tree()?;
        let source_tree = self.source_tree()?;

        let mut ids = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let id_bytes = tile.id.as_bytes();
            let data = serialize(tile)?;

            tiles_tree.insert(id_bytes, data)?;

            // Index by kind
            let kkey = Self::kind_key(&tile.kind, &tile.id);
            kind_tree.insert(kkey, id_bytes)?;

            // Index by source
            if let Some(source) = tile.source {
                let skey = Self::source_key(&source, &tile.id);
                source_tree.insert(skey, id_bytes)?;
            }

            ids.push(tile.id);
        }

        self.db.flush()?;
        Ok(ids)
    }

    /// Retrieve tiles by IDs.
    pub fn retrieve(&self, ids: &[Uuid]) -> Result<Vec<Tile>, Box<dyn std::error::Error>> {
        let tiles_tree = self.tiles_tree()?;
        let mut result = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(data) = tiles_tree.get(id.as_bytes())? {
                result.push(deserialize(&data)?);
            }
        }
        Ok(result)
    }

    /// Compute a simple content hash for search matching.
    fn content_hash(content: &str) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(content.to_lowercase().as_bytes());
        hasher.finalize().to_vec()
    }

    /// Search tiles by query string. Performs content hash matching and embedding similarity.
    pub fn search(
        &self,
        query: &str,
        kind: Option<TileKind>,
        limit: usize,
    ) -> Result<Vec<Tile>, Box<dyn std::error::Error>> {
        let tiles_tree = self.tiles_tree()?;
        let query_lower = query.to_lowercase();

        let mut candidates: Vec<(f32, Tile)> = Vec::new();
        let query_hash = Self::content_hash(query);

        for item in tiles_tree.iter() {
            let (_, data) = item?;
            let tile: Tile = deserialize(&data)?;

            if let Some(ref k) = kind {
                if &tile.kind != k {
                    continue;
                }
            }

            let mut score = 0.0f32;

            // Check if query appears in content (as UTF-8)
            if let Ok(text) = std::str::from_utf8(&tile.content) {
                let text_lower = text.to_lowercase();
                if text_lower.contains(&query_lower) {
                    score += 10.0;
                }
                // Count occurrences
                let count = text_lower.matches(&query_lower).count();
                score += count as f32;
            }

            // Check metadata matches
            for (key, val) in &tile.metadata {
                if val.to_lowercase().contains(&query_lower) || key.to_lowercase().contains(&query_lower) {
                    score += 5.0;
                }
            }

            // Simple embedding dot product if available and non-empty
            if !tile.embedding.is_empty() {
                // Use query hash as pseudo-embedding for deterministic scoring
                let pseudo_emb: Vec<f32> = query_hash.iter()
                    .take(tile.embedding.len().min(query_hash.len()))
                    .map(|&b| (b as f32) / 255.0)
                    .collect();
                let dot: f32 = tile.embedding.iter()
                    .take(pseudo_emb.len())
                    .zip(pseudo_emb.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                score += dot;
            }

            if score > 0.0 {
                candidates.push((score, tile));
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);

        Ok(candidates.into_iter().map(|(_, t)| t).collect())
    }

    /// Retrieve all tiles from a given source.
    pub fn by_source(&self, source: Uuid) -> Result<Vec<Tile>, Box<dyn std::error::Error>> {
        let source_tree = self.source_tree()?;
        let tiles_tree = self.tiles_tree()?;

        let mut result = Vec::new();
        let prefix = source.as_bytes().to_vec();
        for item in source_tree.scan_prefix(&prefix) {
            let (_, id_bytes) = item?;
            if let Some(data) = tiles_tree.get(&id_bytes)? {
                result.push(deserialize(&data)?);
            }
        }
        Ok(result)
    }

    /// Retrieve all tiles of a given kind.
    pub fn by_kind(&self, kind: TileKind) -> Result<Vec<Tile>, Box<dyn std::error::Error>> {
        let kind_tree = self.kind_tree()?;
        let tiles_tree = self.tiles_tree()?;

        let kind_str = match &kind {
            TileKind::Text => "Text",
            TileKind::Code => "Code",
            TileKind::Audio => "Audio",
            TileKind::Image => "Image",
            TileKind::Structured => "Structured",
            TileKind::Unknown(s) => s.as_str(),
        };
        let prefix = format!("{}:", kind_str).into_bytes();

        let mut result = Vec::new();
        for item in kind_tree.scan_prefix(&prefix) {
            let (_, id_bytes) = item?;
            if let Some(data) = tiles_tree.get(&id_bytes)? {
                result.push(deserialize(&data)?);
            }
        }
        Ok(result)
    }

    /// Get store statistics.
    #[allow(clippy::manual_flatten, clippy::unnecessary_sort_by)]
    pub fn stats(&self) -> StoreStats {
        let tiles_tree = self.tiles_tree().ok();
        let total_tiles = tiles_tree.as_ref().map(|t| t.len()).unwrap_or(0) as u64;

        let mut total_bytes = 0u64;
        let mut kind_counts_raw: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

        if let Some(tree) = &tiles_tree {
            for item in tree.iter() {
                if let Ok((_, data)) = item {
                    total_bytes += data.len() as u64;
                    if let Ok(tile) = deserialize::<Tile>(&data) {
                        let kind_str = match tile.kind {
                            TileKind::Text => "Text".into(),
                            TileKind::Code => "Code".into(),
                            TileKind::Audio => "Audio".into(),
                            TileKind::Image => "Image".into(),
                            TileKind::Structured => "Structured".into(),
                            TileKind::Unknown(ref s) => format!("Unknown({})", s),
                        };
                        *kind_counts_raw.entry(kind_str).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut kind_counts: Vec<(String, u64)> = kind_counts_raw.into_iter().collect();
        kind_counts.sort_by(|a, b| b.1.cmp(&a.1));

        StoreStats { total_tiles, total_bytes, kind_counts }
    }

    /// Remove orphaned tiles (tiles whose source no longer exists).
    #[allow(clippy::manual_flatten)]
    pub fn compact(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let tiles_tree = self.tiles_tree()?;
        let source_tree = self.source_tree()?;
        let kind_tree = self.kind_tree()?;

        // Collect all existing IDs
        let all_ids: Vec<Uuid> = {
            let mut ids = Vec::new();
            for item in tiles_tree.iter() {
                if let Ok((key, _)) = item {
                    if let Ok(id) = Uuid::from_slice(&key) {
                        ids.push(id);
                    }
                }
            }
            ids
        };

        // Find tiles whose source doesn't exist
        let existing_set: std::collections::HashSet<Uuid> = all_ids.iter().copied().collect();
        let mut removed = 0u64;

        for id in &all_ids {
            if let Some(data) = tiles_tree.get(id.as_bytes())? {
                let tile: Tile = deserialize(&data)?;
                if let Some(source) = tile.source {
                    if !existing_set.contains(&source) {
                        // Source doesn't exist, remove this tile
                        tiles_tree.remove(id.as_bytes())?;

                        let kkey = Self::kind_key(&tile.kind, id);
                        kind_tree.remove(kkey)?;

                        let skey = Self::source_key(&source, id);
                        source_tree.remove(skey)?;

                        removed += 1;
                    }
                }
            }
        }

        self.db.flush()?;
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tile(kind: TileKind, content: &str, source: Option<Uuid>) -> Tile {
        Tile {
            id: Uuid::new_v4(),
            kind,
            content: content.as_bytes().to_vec(),
            source,
            metadata: vec![],
            embedding: vec![],
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    fn make_tile_with_meta(kind: TileKind, content: &str, source: Option<Uuid>, meta: Vec<(String, String)>) -> Tile {
        let mut t = make_tile(kind, content, source);
        t.metadata = meta;
        t
    }

    #[test]
    fn test_open_and_close() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path());
        assert!(store.is_ok());
        drop(store);
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let tile = make_tile(TileKind::Text, "hello world", None);
        let ids = store.store(&[tile.clone()]).unwrap();
        assert_eq!(ids.len(), 1);

        let retrieved = store.retrieve(&ids).unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].content, tile.content);
        assert_eq!(retrieved[0].id, tile.id);
    }

    #[test]
    fn test_search_by_content() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let t1 = make_tile(TileKind::Text, "the quick brown fox", None);
        let t2 = make_tile(TileKind::Text, "jumps over the lazy dog", None);
        let t3 = make_tile(TileKind::Code, "fn fox() -> i32 { 42 }", None);
        store.store(&[t1, t2, t3]).unwrap();

        let results = store.search("fox", None, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|t| std::str::from_utf8(&t.content).unwrap().contains("fox")));
    }

    #[test]
    fn test_search_with_kind_filter() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let t1 = make_tile(TileKind::Text, "fox in text", None);
        let t2 = make_tile(TileKind::Code, "fox in code", None);
        store.store(&[t1, t2]).unwrap();

        let results = store.search("fox", Some(TileKind::Code), 10).unwrap();
        assert!(results.iter().all(|t| t.kind == TileKind::Code));
    }

    #[test]
    fn test_search_by_metadata() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let t = make_tile_with_meta(
            TileKind::Text,
            "some content",
            None,
            vec![("author".into(), "plato".into())],
        );
        store.store(&[t]).unwrap();

        let results = store.search("plato", None, 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_by_source() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let source_id = Uuid::new_v4();
        let t1 = make_tile(TileKind::Text, "from source", Some(source_id));
        let t2 = make_tile(TileKind::Text, "no source", None);
        store.store(&[t1, t2]).unwrap();

        let results = store.by_source(source_id).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, Some(source_id));
    }

    #[test]
    fn test_by_kind() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let t1 = make_tile(TileKind::Text, "text tile", None);
        let t2 = make_tile(TileKind::Code, "code tile", None);
        let t3 = make_tile(TileKind::Text, "more text", None);
        store.store(&[t1, t2, t3]).unwrap();

        let text_results = store.by_kind(TileKind::Text).unwrap();
        assert_eq!(text_results.len(), 2);
        assert!(text_results.iter().all(|t| t.kind == TileKind::Text));

        let code_results = store.by_kind(TileKind::Code).unwrap();
        assert_eq!(code_results.len(), 1);
    }

    #[test]
    fn test_stats() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let t1 = make_tile(TileKind::Text, "hello", None);
        let t2 = make_tile(TileKind::Code, "fn main() {}", None);
        store.store(&[t1, t2]).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_tiles, 2);
        assert!(stats.total_bytes > 0);
        assert!(stats.kind_counts.iter().any(|(k, _)| k == "Text"));
        assert!(stats.kind_counts.iter().any(|(k, _)| k == "Code"));
    }

    #[test]
    fn test_compact_removes_orphans() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        // source_tile has no source itself (source: None)
        // child_tile points to a UUID that is NOT in the store at all
        let fake_source = Uuid::new_v4();
        let child_tile = make_tile(TileKind::Text, "child", Some(fake_source));
        store.store(&[child_tile]).unwrap();

        // child_tile's source (fake_source) doesn't exist in the store
        let removed = store.compact().unwrap();
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_compact_keeps_non_orphans() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let source_tile = make_tile(TileKind::Text, "parent", None);
        let child_tile = make_tile(TileKind::Text, "child", Some(source_tile.id));
        store.store(&[source_tile.clone(), child_tile]).unwrap();

        let removed = store.compact().unwrap();
        assert_eq!(removed, 0);

        let retrieved = store.retrieve(&[source_tile.id]).unwrap();
        assert_eq!(retrieved.len(), 1);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(TileStore::open(dir.path()).unwrap());

        let mut handles = Vec::new();
        for i in 0..4 {
            let s = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let tile = make_tile(TileKind::Text, &format!("concurrent {}", i), None);
                s.store(&[tile]).unwrap()
            }));
        }

        for h in handles {
            assert!(h.join().unwrap().len() == 1);
        }

        let stats = store.stats();
        assert_eq!(stats.total_tiles, 4);
    }

    #[test]
    fn test_large_batch() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let tiles: Vec<Tile> = (0..500)
            .map(|i| make_tile(TileKind::Text, &format!("tile content {}", i), None))
            .collect();

        let ids = store.store(&tiles).unwrap();
        assert_eq!(ids.len(), 500);

        let retrieved = store.retrieve(&ids).unwrap();
        assert_eq!(retrieved.len(), 500);

        let stats = store.stats();
        assert_eq!(stats.total_tiles, 500);
    }

    #[test]
    fn test_retrieve_nonexistent() {
        let dir = TempDir::new().unwrap();
        let store = TileStore::open(dir.path()).unwrap();

        let results = store.retrieve(&[Uuid::new_v4()]).unwrap();
        assert!(results.is_empty());
    }
}
