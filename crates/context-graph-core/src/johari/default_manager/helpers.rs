//! Internal helper functions for DefaultJohariManager.

use crate::types::fingerprint::{JohariFingerprint, NUM_EMBEDDERS};
use crate::types::JohariQuadrant;

use super::super::manager::QuadrantPattern;

/// Set quadrant weights based on the quadrant.
///
/// Sets 100% weight to the specified quadrant (hard classification).
pub(crate) fn set_quadrant_weights(
    johari: &mut JohariFingerprint,
    embedder_idx: usize,
    quadrant: JohariQuadrant,
) {
    match quadrant {
        JohariQuadrant::Open => johari.set_quadrant(embedder_idx, 1.0, 0.0, 0.0, 0.0, 1.0),
        JohariQuadrant::Hidden => johari.set_quadrant(embedder_idx, 0.0, 1.0, 0.0, 0.0, 1.0),
        JohariQuadrant::Blind => johari.set_quadrant(embedder_idx, 0.0, 0.0, 1.0, 0.0, 1.0),
        JohariQuadrant::Unknown => johari.set_quadrant(embedder_idx, 0.0, 0.0, 0.0, 1.0, 1.0),
    }
}

/// Check if a JohariFingerprint matches a QuadrantPattern.
pub(crate) fn matches_pattern(johari: &JohariFingerprint, pattern: &QuadrantPattern) -> bool {
    match pattern {
        QuadrantPattern::AllIn(target) => {
            (0..NUM_EMBEDDERS).all(|i| johari.dominant_quadrant(i) == *target)
        }
        QuadrantPattern::AtLeast { quadrant, count } => {
            (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == *quadrant)
                .count()
                >= *count
        }
        QuadrantPattern::Exact(expected) => {
            (0..NUM_EMBEDDERS).all(|i| johari.dominant_quadrant(i) == expected[i])
        }
        QuadrantPattern::Mixed {
            min_open,
            max_unknown,
        } => {
            let open_count = (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == JohariQuadrant::Open)
                .count();
            let unknown_count = (0..NUM_EMBEDDERS)
                .filter(|&i| johari.dominant_quadrant(i) == JohariQuadrant::Unknown)
                .count();
            open_count >= *min_open && unknown_count <= *max_unknown
        }
    }
}
