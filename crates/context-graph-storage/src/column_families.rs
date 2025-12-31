//! RocksDB column family definitions.
//!
//! Column families provide logical separation of data types and
//! enable efficient range queries within each category.
//!
//! # Column Families
//! | Name | Purpose | Key Format |
//! |------|---------|------------|
//! | nodes | Primary node storage | NodeId (UUID bytes) |
//! | edges | Graph edge storage | EdgeId (UUID bytes) |
//! | embeddings | Embedding vectors | NodeId (UUID bytes) |
//! | metadata | Node metadata | NodeId (UUID bytes) |
//! | johari_open | Open quadrant index | NodeId |
//! | johari_hidden | Hidden quadrant index | NodeId |
//! | johari_blind | Blind quadrant index | NodeId |
//! | johari_unknown | Unknown quadrant index | NodeId |
//! | temporal | Time-based index | timestamp_ms:NodeId |
//! | tags | Tag index | tag:NodeId |
//! | sources | Source index | source_uri:NodeId |
//! | system | System metadata | key string |

// TODO: Implement get_column_family_descriptors() in TASK-M02-015
