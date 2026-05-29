# forge-memory

External tile memory store for Plato agents. Part of the **forge-flux** ecosystem.

## Concept

Plato agents operate on **tiles** — discrete chunks of content (text, code, audio, images) that flow through a processing pipeline. `forge-memory` provides the persistent storage layer where tiles live between processing stages.

Think of it as an agent's **hippocampus** — tiles go in, get indexed by kind and source, and can be retrieved by ID, searched by content, or queried by metadata.

## Features

- **Embedded storage** via [sled](https://github.com/spacejam/sled) — zero network dependencies, no external database
- **Bincode serialization** — fast, compact binary encoding for tiles
- **Content search** — full-text search with optional kind filtering and embedding similarity
- **Index-based queries** — look up tiles by source UUID or kind
- **Compact** — garbage-collect orphaned tiles whose sources no longer exist
- **Thread-safe** — concurrent reads/writes via sled's lock-free design

## Usage

```rust
use forge_memory::{TileStore, Tile, TileKind};
use uuid::Uuid;

let store = TileStore::open(path)?;

let tile = Tile {
    id: Uuid::new_v4(),
    kind: TileKind::Text,
    content: b"Hello, Plato!".to_vec(),
    source: None,
    metadata: vec![("author".into(), "agent".into())],
    embedding: vec![],
    created_at: 0,
};

let ids = store.store(&[tile])?;
let results = store.search("hello", None, 10)?;
let stats = store.stats();
```

## How It Feeds Plato Agents

1. **Ingest**: Raw content gets decomposed by `forge-code`, `forge-soniqo`, or similar decomposers
2. **Store**: Resulting tiles are persisted in `forge-memory` with kind and source indices
3. **Retrieve**: Downstream agents query tiles by kind, source, or content
4. **Search**: Agents can semantically search the tile space for relevant context
5. **Compact**: Periodic garbage collection removes orphaned tiles

## License

MIT
