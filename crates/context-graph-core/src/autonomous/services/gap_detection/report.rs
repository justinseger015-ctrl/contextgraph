//! Gap detection report types.

use std::collections::HashMap;

use super::types::GapType;

/// Report of detected gaps with analysis
#[derive(Clone, Debug)]
pub struct GapReport {
    /// All detected gaps
    pub gaps: Vec<GapType>,
    /// Overall coverage score (0.0-1.0)
    pub coverage_score: f32,
    /// Generated recommendations for addressing gaps
    pub recommendations: Vec<String>,
    /// Count of goals analyzed
    pub goals_analyzed: usize,
    /// Count of domains detected
    pub domains_detected: usize,
}

impl GapReport {
    /// Check if there are any gaps
    pub fn has_gaps(&self) -> bool {
        !self.gaps.is_empty()
    }

    /// Get gaps sorted by severity (highest first)
    pub fn gaps_by_severity(&self) -> Vec<&GapType> {
        let mut sorted: Vec<_> = self.gaps.iter().collect();
        sorted.sort_by(|a, b| {
            b.severity()
                .partial_cmp(&a.severity())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Count gaps by type
    pub fn gap_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for gap in &self.gaps {
            let type_name = match gap {
                GapType::UncoveredDomain { .. } => "uncovered_domain",
                GapType::WeakCoverage { .. } => "weak_coverage",
                GapType::MissingLink { .. } => "missing_link",
                GapType::TemporalGap { .. } => "temporal_gap",
            };
            *counts.entry(type_name.to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Get the most severe gap, if any
    pub fn most_severe_gap(&self) -> Option<&GapType> {
        self.gaps.iter().max_by(|a, b| {
            a.severity()
                .partial_cmp(&b.severity())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}
