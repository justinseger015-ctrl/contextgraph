//! Test helper functions for PruningService tests

use chrono::{Duration, Utc};

use crate::autonomous::curation::MemoryId;
use crate::autonomous::services::pruning_service::types::MemoryMetadata;

/// Helper to create test metadata
pub fn make_metadata(
    alignment: f32,
    age_days: i64,
    connections: u32,
    byte_size: u64,
) -> MemoryMetadata {
    let created_at = Utc::now() - Duration::days(age_days);
    let mut meta = MemoryMetadata::new(MemoryId::new(), created_at, alignment);
    meta.connection_count = connections;
    meta.byte_size = byte_size;
    meta
}

pub fn make_metadata_with_access(
    alignment: f32,
    age_days: i64,
    connections: u32,
    days_since_access: i64,
) -> MemoryMetadata {
    let created_at = Utc::now() - Duration::days(age_days);
    let mut meta = MemoryMetadata::new(MemoryId::new(), created_at, alignment);
    meta.connection_count = connections;
    meta.last_accessed = Some(Utc::now() - Duration::days(days_since_access));
    meta
}
