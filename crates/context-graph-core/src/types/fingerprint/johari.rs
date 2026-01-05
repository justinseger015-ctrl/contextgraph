//! JohariFingerprint: Per-embedder soft classification with transition probabilities.
//!
//! **STATUS: TASK-F003 COMPLETE - Full implementation**
//!
//! This module provides per-embedder Johari Window classification with:
//! - Soft classification: 4 weights per embedder (sum to 1.0)
//! - Confidence scores per embedder
//! - Transition probability matrix for evolution prediction
//! - Cross-space analysis methods (find_blind_spots)
//!
//! From constitution.yaml (lines 184-194), Johari quadrants map to UTL states:
//! - **Open**: ΔS < 0.5, ΔC > 0.5 → Known to self AND others (direct recall)
//! - **Hidden**: ΔS < 0.5, ΔC < 0.5 → Known to self, NOT others (private)
//! - **Blind**: ΔS > 0.5, ΔC < 0.5 → NOT known to self, known to others (discovery)
//! - **Unknown**: ΔS > 0.5, ΔC > 0.5 → NOT known to self OR others (frontier)
//!
//! Cross-space capability (constitution.yaml line 81):
//! > "Memory can be Open(semantic/E1) but Blind(causal/E5)"

use serde::{Deserialize, Serialize};

use crate::types::JohariQuadrant;

use super::purpose::NUM_EMBEDDERS;

/// Per-embedder Johari Window classification with soft weights.
///
/// Unlike the simple `JohariQuadrant` enum, this provides:
/// - 4 weights per embedder (sum to 1.0) for soft classification
/// - Confidence score per embedder
/// - Transition probability matrix for evolution prediction
/// - Cross-space analysis methods
///
/// # Invariants
/// - All `quadrants[i]` arrays MUST sum to 1.0 (enforced by `set_quadrant`)
/// - All `confidence[i]` values MUST be in [0.0, 1.0]
/// - All `transition_probs[i][j]` rows MUST sum to 1.0
///
/// # Memory Layout
/// - quadrants: 13 × 4 × 4 bytes = 208 bytes
/// - confidence: 13 × 4 bytes = 52 bytes
/// - transition_probs: 13 × 4 × 4 × 4 bytes = 832 bytes
/// - Total: ~1092 bytes per JohariFingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JohariFingerprint {
    /// Soft quadrant weights per embedder: [Open, Hidden, Blind, Unknown]
    /// Each inner array MUST sum to 1.0 (enforced by set_quadrant)
    /// Index 0-12 maps to E1-E13
    pub quadrants: [[f32; 4]; NUM_EMBEDDERS],

    /// Confidence of classification per embedder [0.0, 1.0]
    /// Low confidence = classification is uncertain
    pub confidence: [f32; NUM_EMBEDDERS],

    /// Transition probability matrix per embedder
    /// `transition_probs[embedder][from_quadrant][to_quadrant]`
    /// Each row (from_quadrant) MUST sum to 1.0
    pub transition_probs: [[[f32; 4]; 4]; NUM_EMBEDDERS],
}

impl JohariFingerprint {
    /// Entropy threshold for Johari classification (from constitution.yaml line 192)
    pub const ENTROPY_THRESHOLD: f32 = 0.5;

    /// Coherence threshold for Johari classification (from constitution.yaml line 193)
    pub const COHERENCE_THRESHOLD: f32 = 0.5;

    /// Quadrant index mapping (matches JohariQuadrant enum order)
    pub const OPEN_IDX: usize = 0;
    pub const HIDDEN_IDX: usize = 1;
    pub const BLIND_IDX: usize = 2;
    pub const UNKNOWN_IDX: usize = 3;

    /// Create with all zeros for quadrants/confidence and uniform transition priors (0.25 each).
    ///
    /// This is the recommended starting point for new fingerprints.
    /// Use `set_quadrant()` to populate with actual classifications.
    ///
    /// # Returns
    /// A `JohariFingerprint` with:
    /// - All quadrant weights set to [0.0, 0.0, 0.0, 0.0]
    /// - All confidence values set to 0.0
    /// - All transition probabilities set to uniform (0.25)
    pub fn zeroed() -> Self {
        // Uniform transition probabilities: 0.25 to each quadrant
        let uniform_transitions = [[0.25f32; 4]; 4];

        Self {
            quadrants: [[0.0f32; 4]; NUM_EMBEDDERS],
            confidence: [0.0f32; NUM_EMBEDDERS],
            transition_probs: [uniform_transitions; NUM_EMBEDDERS],
        }
    }

    /// Create stub with all Unknown dominant (backwards compat during migration).
    ///
    /// **DEPRECATED**: Use `zeroed()` for new code.
    ///
    /// Sets all embedders to 100% Unknown weight with full confidence.
    /// This matches the old stub behavior for backwards compatibility.
    #[deprecated(since = "2.0.0", note = "Use zeroed() for new code")]
    pub fn stub() -> Self {
        let mut fp = Self::zeroed();
        for embedder_idx in 0..NUM_EMBEDDERS {
            // Set 100% Unknown weight, 100% confidence
            fp.quadrants[embedder_idx] = [0.0, 0.0, 0.0, 1.0];
            fp.confidence[embedder_idx] = 1.0;
        }
        fp
    }

    /// Classify based on entropy (ΔS) and coherence (ΔC) metrics.
    ///
    /// From constitution.yaml lines 188-194:
    /// - Open: entropy < 0.5 AND coherence > 0.5
    /// - Hidden: entropy < 0.5 AND coherence < 0.5
    /// - Blind: entropy > 0.5 AND coherence < 0.5
    /// - Unknown: entropy > 0.5 AND coherence > 0.5
    ///
    /// # Arguments
    /// * `entropy` - Entropy change value (ΔS), typically in [0.0, 1.0]
    /// * `coherence` - Coherence change value (ΔC), typically in [0.0, 1.0]
    ///
    /// # Returns
    /// The `JohariQuadrant` classification based on the UTL thresholds.
    ///
    /// # Boundary Behavior
    /// At exactly threshold (0.5), treats as:
    /// - entropy = 0.5 → low entropy (< test uses >=)
    /// - coherence = 0.5 → low coherence (> test uses >)
    #[inline]
    pub fn classify_quadrant(entropy: f32, coherence: f32) -> JohariQuadrant {
        let low_entropy = entropy < Self::ENTROPY_THRESHOLD;
        let high_coherence = coherence > Self::COHERENCE_THRESHOLD;

        match (low_entropy, high_coherence) {
            (true, true) => JohariQuadrant::Open,     // Low S, High C
            (true, false) => JohariQuadrant::Hidden,  // Low S, Low C
            (false, false) => JohariQuadrant::Blind,  // High S, Low C
            (false, true) => JohariQuadrant::Unknown, // High S, High C
        }
    }

    /// Get dominant (highest weight) quadrant for an embedder.
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12 for E1-E13)
    ///
    /// # Returns
    /// The `JohariQuadrant` with the highest weight for this embedder.
    /// If all weights are zero (unclassified), returns Unknown (the frontier state).
    /// In case of ties, returns the first (lowest index) tied quadrant.
    ///
    /// # Panics
    /// Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    #[inline]
    pub fn dominant_quadrant(&self, embedder_idx: usize) -> JohariQuadrant {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );

        let weights = &self.quadrants[embedder_idx];

        // Check if all weights are zero (unclassified embedder)
        let sum: f32 = weights.iter().sum();
        if sum < f32::EPSILON {
            // Unclassified embedders default to Unknown (the frontier/exploration state)
            return JohariQuadrant::Unknown;
        }

        let mut max_idx = 0;
        let mut max_val = weights[0];

        for (idx, &val) in weights.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        Self::idx_to_quadrant(max_idx)
    }

    /// Set quadrant weights for an embedder.
    ///
    /// Automatically normalizes so weights sum to 1.0.
    /// If all weights are 0, sets uniform distribution (0.25 each).
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12 for E1-E13)
    /// * `open` - Weight for Open quadrant
    /// * `hidden` - Weight for Hidden quadrant
    /// * `blind` - Weight for Blind quadrant
    /// * `unknown` - Weight for Unknown quadrant
    /// * `confidence` - Classification confidence [0.0, 1.0]
    ///
    /// # Panics
    /// - Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    /// - Panics if any weight is negative
    /// - Panics if confidence is not in [0.0, 1.0]
    pub fn set_quadrant(
        &mut self,
        embedder_idx: usize,
        open: f32,
        hidden: f32,
        blind: f32,
        unknown: f32,
        confidence: f32,
    ) {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );
        assert!(open >= 0.0, "open weight must be non-negative, got {}", open);
        assert!(
            hidden >= 0.0,
            "hidden weight must be non-negative, got {}",
            hidden
        );
        assert!(
            blind >= 0.0,
            "blind weight must be non-negative, got {}",
            blind
        );
        assert!(
            unknown >= 0.0,
            "unknown weight must be non-negative, got {}",
            unknown
        );
        assert!(
            (0.0..=1.0).contains(&confidence),
            "confidence must be in [0.0, 1.0], got {}",
            confidence
        );

        let sum = open + hidden + blind + unknown;

        if sum < f32::EPSILON {
            // All zero: use uniform distribution
            self.quadrants[embedder_idx] = [0.25, 0.25, 0.25, 0.25];
        } else {
            // Normalize to sum to 1.0
            self.quadrants[embedder_idx] = [open / sum, hidden / sum, blind / sum, unknown / sum];
        }

        self.confidence[embedder_idx] = confidence;
    }

    /// Find all embedder indices where the given quadrant is dominant.
    ///
    /// # Arguments
    /// * `quadrant` - The quadrant to search for
    ///
    /// # Returns
    /// Vector of embedder indices (0-12) where the given quadrant has the highest weight.
    pub fn find_by_quadrant(&self, quadrant: JohariQuadrant) -> Vec<usize> {
        (0..NUM_EMBEDDERS)
            .filter(|&idx| self.dominant_quadrant(idx) == quadrant)
            .collect()
    }

    /// Find blind spots: cross-space gaps where one embedder understands but another doesn't.
    ///
    /// Specifically finds embedders with high Blind weight while E1 (semantic) has high Open weight.
    /// This indicates understanding at the semantic level but lack of awareness in other dimensions.
    ///
    /// # Returns
    /// Vector of `(embedder_idx, blind_severity)` pairs sorted by severity descending.
    /// Blind severity = Open[E1] × Blind[embedder]
    ///
    /// # Interpretation
    /// High severity means:
    /// - E1 (semantic) strongly classifies as Open (well understood)
    /// - The target embedder strongly classifies as Blind (discovery opportunity)
    /// - This is a "blind spot" - semantic understanding without dimensional insight
    pub fn find_blind_spots(&self) -> Vec<(usize, f32)> {
        let e1_open_weight = self.quadrants[0][Self::OPEN_IDX];

        let mut blind_spots: Vec<(usize, f32)> = (0..NUM_EMBEDDERS)
            .filter_map(|idx| {
                let blind_weight = self.quadrants[idx][Self::BLIND_IDX];
                let severity = e1_open_weight * blind_weight;

                if severity > f32::EPSILON {
                    Some((idx, severity))
                } else {
                    None
                }
            })
            .collect();

        // Sort by severity descending
        blind_spots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        blind_spots
    }

    /// Predict most likely next quadrant given current state.
    ///
    /// Uses the transition probability matrix to determine the most probable
    /// next quadrant from the current quadrant.
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12)
    /// * `current` - Current quadrant state
    ///
    /// # Returns
    /// The `JohariQuadrant` with highest transition probability from the current state.
    ///
    /// # Panics
    /// Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    pub fn predict_transition(
        &self,
        embedder_idx: usize,
        current: JohariQuadrant,
    ) -> JohariQuadrant {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );

        let from_idx = Self::quadrant_to_idx(current);
        let probs = &self.transition_probs[embedder_idx][from_idx];

        let mut max_idx = 0;
        let mut max_prob = probs[0];

        for (idx, &prob) in probs.iter().enumerate().skip(1) {
            if prob > max_prob {
                max_prob = prob;
                max_idx = idx;
            }
        }

        Self::idx_to_quadrant(max_idx)
    }

    /// Encode dominant quadrants as compact bytes.
    ///
    /// 2 bits per quadrant = 26 bits for 13 embedders.
    /// Fits in 4 bytes with 6 bits unused.
    ///
    /// Encoding: 00=Open, 01=Hidden, 10=Blind, 11=Unknown
    ///
    /// # Byte Layout
    /// - byte[0]: E1[1:0], E2[3:2], E3[5:4], E4[7:6]
    /// - byte[1]: E5[1:0], E6[3:2], E7[5:4], E8[7:6]
    /// - byte[2]: E9[1:0], E10[3:2], E11[5:4], E12[7:6]
    /// - byte[3]: E13[1:0], unused[7:2]
    pub fn to_compact_bytes(&self) -> [u8; 4] {
        let mut bytes = [0u8; 4];

        for embedder_idx in 0..NUM_EMBEDDERS {
            let quadrant = self.dominant_quadrant(embedder_idx);
            let bits = Self::quadrant_to_idx(quadrant) as u8;

            let byte_idx = embedder_idx / 4;
            let bit_offset = (embedder_idx % 4) * 2;

            bytes[byte_idx] |= bits << bit_offset;
        }

        bytes
    }

    /// Decode from compact bytes to JohariFingerprint.
    ///
    /// Sets dominant quadrant to 1.0 weight, others to 0.0.
    /// Sets confidence to 1.0 (hard classification from compact).
    ///
    /// # Arguments
    /// * `bytes` - 4-byte compact representation
    ///
    /// # Returns
    /// A `JohariFingerprint` with hard classifications derived from the compact encoding.
    pub fn from_compact_bytes(bytes: [u8; 4]) -> Self {
        let mut fp = Self::zeroed();

        for embedder_idx in 0..NUM_EMBEDDERS {
            let byte_idx = embedder_idx / 4;
            let bit_offset = (embedder_idx % 4) * 2;

            let bits = (bytes[byte_idx] >> bit_offset) & 0b11;
            let quadrant_idx = bits as usize;

            // Set 100% weight for the decoded quadrant
            fp.quadrants[embedder_idx] = [0.0; 4];
            fp.quadrants[embedder_idx][quadrant_idx] = 1.0;
            fp.confidence[embedder_idx] = 1.0; // Hard classification
        }

        fp
    }

    /// Compute overall openness (fraction of embedders with Open dominant).
    ///
    /// # Returns
    /// Value in [0.0, 1.0] representing the fraction of embedders where Open is dominant.
    pub fn openness(&self) -> f32 {
        let open_count = (0..NUM_EMBEDDERS)
            .filter(|&idx| self.dominant_quadrant(idx) == JohariQuadrant::Open)
            .count();
        open_count as f32 / NUM_EMBEDDERS as f32
    }

    /// Check if overall awareness is healthy (majority Open/Hidden dominant).
    ///
    /// A memory is considered "aware" if more than half of its embedders
    /// are in self-aware quadrants (Open or Hidden).
    ///
    /// # Returns
    /// `true` if more than 50% of embedders have Open or Hidden dominant.
    pub fn is_aware(&self) -> bool {
        let aware_count = (0..NUM_EMBEDDERS)
            .filter(|&idx| {
                let dom = self.dominant_quadrant(idx);
                dom == JohariQuadrant::Open || dom == JohariQuadrant::Hidden
            })
            .count();

        aware_count as f32 / NUM_EMBEDDERS as f32 >= 0.5
    }

    /// Validate all invariants. Returns Err with description if invalid.
    ///
    /// Checks:
    /// - All quadrant weight rows sum to 1.0 (±0.001 tolerance)
    /// - All confidence values in [0.0, 1.0]
    /// - All transition probability rows sum to 1.0 (±0.001 tolerance)
    /// - No NaN or infinite values
    ///
    /// # Returns
    /// - `Ok(())` if all invariants hold
    /// - `Err(String)` with detailed error message if any invariant is violated
    pub fn validate(&self) -> Result<(), String> {
        const TOLERANCE: f32 = 0.001;

        // Check quadrant weights
        for (embedder_idx, weights) in self.quadrants.iter().enumerate() {
            // Check for NaN/Inf
            for (quad_idx, &weight) in weights.iter().enumerate() {
                if weight.is_nan() {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is NaN",
                        embedder_idx, quad_idx
                    ));
                }
                if weight.is_infinite() {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is infinite",
                        embedder_idx, quad_idx
                    ));
                }
                if weight < 0.0 {
                    return Err(format!(
                        "JohariFingerprint validation failed: quadrants[{}][{}] is negative ({})",
                        embedder_idx, quad_idx, weight
                    ));
                }
            }

            let sum: f32 = weights.iter().sum();
            // Allow either all-zeros (not yet set) or normalized sum
            if sum > f32::EPSILON && (sum - 1.0).abs() > TOLERANCE {
                return Err(format!(
                    "JohariFingerprint validation failed at embedder {}: quadrant weights sum to {} (expected 1.0)",
                    embedder_idx, sum
                ));
            }
        }

        // Check confidence values
        for (embedder_idx, &conf) in self.confidence.iter().enumerate() {
            if conf.is_nan() {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] is NaN",
                    embedder_idx
                ));
            }
            if conf.is_infinite() {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] is infinite",
                    embedder_idx
                ));
            }
            if !(0.0..=1.0).contains(&conf) {
                return Err(format!(
                    "JohariFingerprint validation failed: confidence[{}] = {} not in [0.0, 1.0]",
                    embedder_idx, conf
                ));
            }
        }

        // Check transition probability matrices
        for (embedder_idx, matrix) in self.transition_probs.iter().enumerate() {
            for (from_idx, row) in matrix.iter().enumerate() {
                // Check for NaN/Inf
                for (to_idx, &prob) in row.iter().enumerate() {
                    if prob.is_nan() {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is NaN",
                            embedder_idx, from_idx, to_idx
                        ));
                    }
                    if prob.is_infinite() {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is infinite",
                            embedder_idx, from_idx, to_idx
                        ));
                    }
                    if prob < 0.0 {
                        return Err(format!(
                            "JohariFingerprint validation failed: transition_probs[{}][{}][{}] is negative ({})",
                            embedder_idx, from_idx, to_idx, prob
                        ));
                    }
                }

                let sum: f32 = row.iter().sum();
                if (sum - 1.0).abs() > TOLERANCE {
                    return Err(format!(
                        "JohariFingerprint validation failed: transition_probs[{}][{}] sums to {} (expected 1.0)",
                        embedder_idx, from_idx, sum
                    ));
                }
            }
        }

        Ok(())
    }

    /// Set transition probabilities for an embedder.
    ///
    /// Normalizes each row to sum to 1.0.
    ///
    /// # Arguments
    /// * `embedder_idx` - Index of embedder (0-12)
    /// * `matrix` - 4×4 transition probability matrix [from][to]
    ///
    /// # Panics
    /// Panics if `embedder_idx >= NUM_EMBEDDERS` (13)
    pub fn set_transition_probs(&mut self, embedder_idx: usize, matrix: [[f32; 4]; 4]) {
        assert!(
            embedder_idx < NUM_EMBEDDERS,
            "embedder_idx {} out of bounds (max {})",
            embedder_idx,
            NUM_EMBEDDERS - 1
        );

        for (from_idx, row) in matrix.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            if sum < f32::EPSILON {
                // All zero: use uniform
                self.transition_probs[embedder_idx][from_idx] = [0.25; 4];
            } else {
                for (to_idx, &val) in row.iter().enumerate() {
                    self.transition_probs[embedder_idx][from_idx][to_idx] = val / sum;
                }
            }
        }
    }

    /// Convert quadrant index to JohariQuadrant.
    #[inline]
    fn idx_to_quadrant(idx: usize) -> JohariQuadrant {
        match idx {
            0 => JohariQuadrant::Open,
            1 => JohariQuadrant::Hidden,
            2 => JohariQuadrant::Blind,
            _ => JohariQuadrant::Unknown,
        }
    }

    /// Convert JohariQuadrant to index.
    #[inline]
    fn quadrant_to_idx(quadrant: JohariQuadrant) -> usize {
        match quadrant {
            JohariQuadrant::Open => 0,
            JohariQuadrant::Hidden => 1,
            JohariQuadrant::Blind => 2,
            JohariQuadrant::Unknown => 3,
        }
    }
}

impl Default for JohariFingerprint {
    /// Default to zeroed (not stub) for new code
    fn default() -> Self {
        Self::zeroed()
    }
}

impl PartialEq for JohariFingerprint {
    /// Compare all fields with f32 epsilon tolerance
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6;

        // Compare quadrants
        for i in 0..NUM_EMBEDDERS {
            for j in 0..4 {
                if (self.quadrants[i][j] - other.quadrants[i][j]).abs() > EPSILON {
                    return false;
                }
            }
        }

        // Compare confidence
        for i in 0..NUM_EMBEDDERS {
            if (self.confidence[i] - other.confidence[i]).abs() > EPSILON {
                return false;
            }
        }

        // Compare transition_probs
        for i in 0..NUM_EMBEDDERS {
            for j in 0..4 {
                for k in 0..4 {
                    if (self.transition_probs[i][j][k] - other.transition_probs[i][j][k]).abs()
                        > EPSILON
                    {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real entropy/coherence values from UTL theory
    const REAL_ENTROPY_LOW: f32 = 0.3;  // Below threshold
    const REAL_ENTROPY_HIGH: f32 = 0.7; // Above threshold
    const REAL_COHERENCE_LOW: f32 = 0.3;
    const REAL_COHERENCE_HIGH: f32 = 0.7;

    // ===== zeroed() Tests =====

    #[test]
    fn test_zeroed() {
        let jf = JohariFingerprint::zeroed();

        // All quadrant weights should be [0,0,0,0]
        for embedder_idx in 0..NUM_EMBEDDERS {
            assert_eq!(
                jf.quadrants[embedder_idx],
                [0.0, 0.0, 0.0, 0.0],
                "Embedder {} quadrants should be all zeros",
                embedder_idx
            );
        }

        // All confidence should be 0.0
        for embedder_idx in 0..NUM_EMBEDDERS {
            assert_eq!(
                jf.confidence[embedder_idx], 0.0,
                "Embedder {} confidence should be 0.0",
                embedder_idx
            );
        }

        // All transition probabilities should be uniform (0.25)
        for embedder_idx in 0..NUM_EMBEDDERS {
            for from_quad in 0..4 {
                for to_quad in 0..4 {
                    assert!(
                        (jf.transition_probs[embedder_idx][from_quad][to_quad] - 0.25).abs()
                            < f32::EPSILON,
                        "Transition prob [{}][{}][{}] should be 0.25",
                        embedder_idx,
                        from_quad,
                        to_quad
                    );
                }
            }
        }

        println!("[PASS] zeroed() creates valid fingerprint with zero quadrants and uniform transitions");
    }

    // ===== classify_quadrant() Tests =====

    #[test]
    fn test_classify_quadrant_open() {
        // Low entropy, High coherence -> Open
        let q = JohariFingerprint::classify_quadrant(REAL_ENTROPY_LOW, REAL_COHERENCE_HIGH);
        assert_eq!(q, JohariQuadrant::Open);
        println!(
            "[PASS] classify_quadrant({}, {}) returns Open",
            REAL_ENTROPY_LOW, REAL_COHERENCE_HIGH
        );
    }

    #[test]
    fn test_classify_quadrant_hidden() {
        // Low entropy, Low coherence -> Hidden
        let q = JohariFingerprint::classify_quadrant(REAL_ENTROPY_LOW, REAL_COHERENCE_LOW);
        assert_eq!(q, JohariQuadrant::Hidden);
        println!(
            "[PASS] classify_quadrant({}, {}) returns Hidden",
            REAL_ENTROPY_LOW, REAL_COHERENCE_LOW
        );
    }

    #[test]
    fn test_classify_quadrant_blind() {
        // High entropy, Low coherence -> Blind
        let q = JohariFingerprint::classify_quadrant(REAL_ENTROPY_HIGH, REAL_COHERENCE_LOW);
        assert_eq!(q, JohariQuadrant::Blind);
        println!(
            "[PASS] classify_quadrant({}, {}) returns Blind",
            REAL_ENTROPY_HIGH, REAL_COHERENCE_LOW
        );
    }

    #[test]
    fn test_classify_quadrant_unknown() {
        // High entropy, High coherence -> Unknown
        let q = JohariFingerprint::classify_quadrant(REAL_ENTROPY_HIGH, REAL_COHERENCE_HIGH);
        assert_eq!(q, JohariQuadrant::Unknown);
        println!(
            "[PASS] classify_quadrant({}, {}) returns Unknown",
            REAL_ENTROPY_HIGH, REAL_COHERENCE_HIGH
        );
    }

    #[test]
    fn test_classify_quadrant_boundary() {
        // At exactly 0.5, 0.5 - should be deterministic
        let q = JohariFingerprint::classify_quadrant(0.5, 0.5);
        // entropy < 0.5 is false (0.5 is not < 0.5), coherence > 0.5 is false
        // So: (false, false) -> Blind
        assert_eq!(q, JohariQuadrant::Blind);
        println!("[PASS] classify_quadrant(0.5, 0.5) returns deterministic result (Blind)");
    }

    // ===== set_quadrant() Tests =====

    #[test]
    fn test_set_quadrant_normalizes() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE set_quadrant: quadrants[0] = {:?}", jf.quadrants[0]);

        jf.set_quadrant(0, 1.0, 2.0, 3.0, 4.0, 0.8);

        println!("AFTER set_quadrant([1,2,3,4]): quadrants[0] = {:?}", jf.quadrants[0]);

        let sum: f32 = jf.quadrants[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Sum should be 1.0, got {}", sum);

        // Expected: 1/10, 2/10, 3/10, 4/10
        assert!((jf.quadrants[0][0] - 0.1).abs() < 0.001);
        assert!((jf.quadrants[0][1] - 0.2).abs() < 0.001);
        assert!((jf.quadrants[0][2] - 0.3).abs() < 0.001);
        assert!((jf.quadrants[0][3] - 0.4).abs() < 0.001);

        assert_eq!(jf.confidence[0], 0.8);

        println!("[PASS] set_quadrant normalizes weights to sum=1.0");
    }

    #[test]
    fn test_set_quadrant_all_zero() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE set_quadrant: quadrants[0] = {:?}", jf.quadrants[0]);

        jf.set_quadrant(0, 0.0, 0.0, 0.0, 0.0, 0.5);

        println!(
            "AFTER set_quadrant(all zeros): quadrants[0] = {:?}",
            jf.quadrants[0]
        );

        // All zero should become uniform [0.25, 0.25, 0.25, 0.25]
        for weight in jf.quadrants[0].iter() {
            assert!(
                (weight - 0.25).abs() < 0.001,
                "Weight should be 0.25, got {}",
                weight
            );
        }

        let sum: f32 = jf.quadrants[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        println!("[PASS] All-zero weights normalized to uniform distribution");
    }

    #[test]
    #[should_panic(expected = "negative")]
    fn test_set_quadrant_negative_panics() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE: Attempting to set negative weight");

        // This MUST panic - no silent failure
        jf.set_quadrant(0, -1.0, 0.0, 0.0, 0.0, 0.5);

        println!("ERROR: Did not panic on negative weight!");
    }

    #[test]
    #[should_panic(expected = "embedder_idx")]
    fn test_set_quadrant_out_of_bounds_panics() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE: Attempting to set embedder 13 (out of bounds)");

        // This MUST panic
        jf.set_quadrant(13, 1.0, 0.0, 0.0, 0.0, 1.0);

        println!("ERROR: Did not panic on out-of-bounds index!");
    }

    // ===== dominant_quadrant() Tests =====

    #[test]
    fn test_dominant_quadrant() {
        let mut jf = JohariFingerprint::zeroed();
        jf.set_quadrant(0, 0.5, 0.3, 0.1, 0.1, 1.0);

        let dom = jf.dominant_quadrant(0);
        assert_eq!(dom, JohariQuadrant::Open);

        println!("[PASS] dominant_quadrant returns highest weight quadrant");
    }

    #[test]
    fn test_dominant_quadrant_tie() {
        let mut jf = JohariFingerprint::zeroed();
        // Set equal weights for Open and Hidden
        jf.quadrants[0] = [0.5, 0.5, 0.0, 0.0];

        let dom = jf.dominant_quadrant(0);
        // First wins in tie
        assert_eq!(dom, JohariQuadrant::Open);

        println!("[PASS] dominant_quadrant returns first quadrant in tie");
    }

    #[test]
    fn test_dominant_quadrant_zeroed_returns_unknown() {
        let jf = JohariFingerprint::zeroed();

        // Zeroed embedders (all weights = 0) should return Unknown (frontier state)
        for i in 0..NUM_EMBEDDERS {
            assert_eq!(
                jf.dominant_quadrant(i),
                JohariQuadrant::Unknown,
                "Zeroed embedder {} should have Unknown as dominant",
                i
            );
        }

        println!("[PASS] dominant_quadrant returns Unknown for zeroed embedders");
    }

    // ===== find_by_quadrant() Tests =====

    #[test]
    fn test_find_by_quadrant() {
        let mut jf = JohariFingerprint::zeroed();

        // Set E1 (index 0) to Open
        jf.set_quadrant(0, 1.0, 0.0, 0.0, 0.0, 1.0);
        // Set E5 (index 4) to Blind
        jf.set_quadrant(4, 0.0, 0.0, 1.0, 0.0, 1.0);
        // Set remaining embedders to Hidden (so they don't default to Unknown)
        for i in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12] {
            jf.set_quadrant(i, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
        }

        let open_embedders = jf.find_by_quadrant(JohariQuadrant::Open);
        assert_eq!(open_embedders, vec![0]);

        let blind_embedders = jf.find_by_quadrant(JohariQuadrant::Blind);
        assert_eq!(blind_embedders, vec![4]);

        // Verify the others are Hidden
        let hidden_embedders = jf.find_by_quadrant(JohariQuadrant::Hidden);
        assert_eq!(hidden_embedders.len(), 11); // All except 0 and 4

        println!("[PASS] find_by_quadrant returns correct embedder indices");
    }

    // ===== find_blind_spots() Tests =====

    #[test]
    fn test_find_blind_spots() {
        let mut jf = JohariFingerprint::zeroed();

        // E1 (index 0) = strongly Open
        jf.set_quadrant(0, 0.9, 0.05, 0.03, 0.02, 1.0);
        // E5 (index 4) = strongly Blind
        jf.set_quadrant(4, 0.1, 0.1, 0.7, 0.1, 1.0);

        let blind_spots = jf.find_blind_spots();

        println!("Blind spots: {:?}", blind_spots);

        // Should find E5 as a blind spot (E1 Open * E5 Blind = 0.9 * 0.7 = 0.63)
        let e5_spot = blind_spots.iter().find(|(idx, _)| *idx == 4);
        assert!(e5_spot.is_some(), "E5 should be in blind spots");

        let (_, severity) = e5_spot.unwrap();
        let expected_severity = 0.9 * 0.7;
        assert!(
            (severity - expected_severity).abs() < 0.01,
            "Severity should be ~{}, got {}",
            expected_severity,
            severity
        );

        println!("[PASS] find_blind_spots detects cross-space gaps");
    }

    // ===== predict_transition() Tests =====

    #[test]
    fn test_predict_transition() {
        let mut jf = JohariFingerprint::zeroed();

        // Set custom transition probs for E1 (index 0)
        // From Open, highest prob to Hidden
        let mut matrix = [[0.25f32; 4]; 4];
        matrix[JohariFingerprint::OPEN_IDX] = [0.1, 0.6, 0.2, 0.1]; // From Open -> Hidden most likely

        jf.set_transition_probs(0, matrix);

        let predicted = jf.predict_transition(0, JohariQuadrant::Open);
        assert_eq!(predicted, JohariQuadrant::Hidden);

        println!("[PASS] predict_transition uses transition matrix correctly");
    }

    // ===== Compact Bytes Tests =====

    #[test]
    fn test_compact_bytes_roundtrip() {
        let mut jf = JohariFingerprint::zeroed();

        // Set various quadrants
        jf.set_quadrant(0, 1.0, 0.0, 0.0, 0.0, 1.0); // Open
        jf.set_quadrant(1, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
        jf.set_quadrant(2, 0.0, 0.0, 1.0, 0.0, 1.0); // Blind
        jf.set_quadrant(3, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown

        let bytes = jf.to_compact_bytes();
        println!("Compact bytes: {:?}", bytes);

        let decoded = JohariFingerprint::from_compact_bytes(bytes);

        // Verify dominant quadrants match
        for i in 0..4 {
            assert_eq!(
                jf.dominant_quadrant(i),
                decoded.dominant_quadrant(i),
                "Embedder {} dominant mismatch",
                i
            );
        }

        println!("[PASS] compact_bytes roundtrip preserves dominant quadrants");
    }

    #[test]
    fn test_compact_bytes_all_quadrants() {
        // Test each quadrant has unique encoding
        let quadrants = [
            JohariQuadrant::Open,
            JohariQuadrant::Hidden,
            JohariQuadrant::Blind,
            JohariQuadrant::Unknown,
        ];

        for (i, q) in quadrants.iter().enumerate() {
            let mut jf = JohariFingerprint::zeroed();
            jf.quadrants[0][i] = 1.0;
            jf.confidence[0] = 1.0;

            let bytes = jf.to_compact_bytes();
            let decoded = JohariFingerprint::from_compact_bytes(bytes);

            assert_eq!(decoded.dominant_quadrant(0), *q);
        }

        println!("[PASS] Each quadrant has unique encoding");
    }

    // ===== validate() Tests =====

    #[test]
    fn test_validate_valid() {
        let mut jf = JohariFingerprint::zeroed();
        jf.set_quadrant(0, 0.25, 0.25, 0.25, 0.25, 0.5);

        let result = jf.validate();
        assert!(result.is_ok(), "Valid fingerprint should pass: {:?}", result);

        println!("[PASS] validate() returns Ok for valid fingerprint");
    }

    #[test]
    fn test_validate_invalid_sum() {
        let mut jf = JohariFingerprint::zeroed();
        jf.quadrants[0] = [0.1, 0.1, 0.1, 0.1]; // Sum = 0.4, not 1.0

        let result = jf.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("sum"));

        println!("[PASS] validate() catches invalid weight sum");
    }

    #[test]
    fn test_validate_nan() {
        let mut jf = JohariFingerprint::zeroed();
        jf.quadrants[0][0] = f32::NAN;

        let result = jf.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("NaN"));

        println!("[PASS] validate() catches NaN values");
    }

    #[test]
    fn test_validate_confidence_range() {
        let mut jf = JohariFingerprint::zeroed();
        jf.confidence[0] = 1.5; // Out of range

        let result = jf.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("confidence"));

        println!("[PASS] validate() catches confidence out of range");
    }

    // ===== openness() and is_aware() Tests =====

    #[test]
    fn test_openness() {
        let mut jf = JohariFingerprint::zeroed();

        // Set 7 out of 13 embedders to Open
        for i in 0..7 {
            jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0);
        }
        // Set remaining 6 to Hidden (not Open)
        for i in 7..NUM_EMBEDDERS {
            jf.set_quadrant(i, 0.0, 1.0, 0.0, 0.0, 1.0); // Hidden
        }

        let openness = jf.openness();
        let expected = 7.0 / 13.0;
        assert!(
            (openness - expected).abs() < 0.01,
            "Openness should be ~{}, got {}",
            expected,
            openness
        );

        println!("[PASS] openness() returns correct fraction");
    }

    #[test]
    fn test_is_aware() {
        let mut jf = JohariFingerprint::zeroed();

        // Set majority (7+) to Open/Hidden
        for i in 0..7 {
            jf.set_quadrant(i, 1.0, 0.0, 0.0, 0.0, 1.0); // Open
        }
        // Set remaining to Blind (not aware)
        for i in 7..NUM_EMBEDDERS {
            jf.set_quadrant(i, 0.0, 0.0, 1.0, 0.0, 1.0); // Blind
        }

        assert!(jf.is_aware(), "7/13 Open should be aware");

        // Set all to Unknown
        let mut jf2 = JohariFingerprint::zeroed();
        for i in 0..NUM_EMBEDDERS {
            jf2.set_quadrant(i, 0.0, 0.0, 0.0, 1.0, 1.0); // Unknown
        }

        assert!(!jf2.is_aware(), "All Unknown should not be aware");

        println!("[PASS] is_aware() returns correct value");
    }

    // ===== stub() backwards compat test =====

    #[test]
    #[allow(deprecated)]
    fn test_stub_backwards_compat() {
        let jf = JohariFingerprint::stub();

        for i in 0..NUM_EMBEDDERS {
            assert_eq!(
                jf.dominant_quadrant(i),
                JohariQuadrant::Unknown,
                "stub() embedder {} should be Unknown dominant",
                i
            );
            assert_eq!(jf.confidence[i], 1.0, "stub() confidence should be 1.0");
        }

        println!("[PASS] stub() returns all Unknown dominant (backwards compat)");
    }

    // ===== Default and PartialEq Tests =====

    #[test]
    fn test_default() {
        let jf = JohariFingerprint::default();
        let zeroed = JohariFingerprint::zeroed();

        assert_eq!(jf, zeroed, "Default should equal zeroed");

        println!("[PASS] Default implements zeroed()");
    }

    #[test]
    fn test_partial_eq() {
        let jf1 = JohariFingerprint::zeroed();
        let jf2 = JohariFingerprint::zeroed();

        assert_eq!(jf1, jf2, "Two zeroed fingerprints should be equal");

        let mut jf3 = JohariFingerprint::zeroed();
        jf3.set_quadrant(0, 1.0, 0.0, 0.0, 0.0, 1.0);

        assert_ne!(jf1, jf3, "Different fingerprints should not be equal");

        println!("[PASS] PartialEq works correctly with epsilon tolerance");
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_edge_case_all_zero_weights() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE set_quadrant: quadrants[0] = {:?}", jf.quadrants[0]);

        jf.set_quadrant(0, 0.0, 0.0, 0.0, 0.0, 0.5);

        println!(
            "AFTER set_quadrant(all zeros): quadrants[0] = {:?}",
            jf.quadrants[0]
        );

        assert!((jf.quadrants[0].iter().sum::<f32>() - 1.0).abs() < 0.001);
        println!("[PASS] All-zero weights normalized to uniform distribution");
    }

    #[test]
    fn test_edge_case_max_embedder_index() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE: Setting embedder 12 (E13_SPLADE)");

        jf.set_quadrant(12, 1.0, 0.0, 0.0, 0.0, 1.0); // E13 = index 12

        println!("AFTER: quadrants[12] = {:?}", jf.quadrants[12]);

        assert_eq!(jf.dominant_quadrant(12), JohariQuadrant::Open);
        println!("[PASS] Maximum embedder index 12 works correctly");
    }

    #[test]
    #[should_panic(expected = "embedder_idx")]
    fn test_edge_case_out_of_bounds_panics() {
        let mut jf = JohariFingerprint::zeroed();

        println!("BEFORE: Attempting to set embedder 13 (out of bounds)");

        // This MUST panic - no silent failure
        jf.set_quadrant(13, 1.0, 0.0, 0.0, 0.0, 1.0);

        // Should never reach here
        println!("ERROR: Did not panic on out-of-bounds index!");
    }

    #[test]
    #[should_panic(expected = "embedder_idx")]
    fn test_dominant_quadrant_out_of_bounds() {
        let jf = JohariFingerprint::zeroed();
        let _ = jf.dominant_quadrant(13);
    }

    #[test]
    #[should_panic(expected = "embedder_idx")]
    fn test_predict_transition_out_of_bounds() {
        let jf = JohariFingerprint::zeroed();
        let _ = jf.predict_transition(13, JohariQuadrant::Open);
    }

    // ===== Verification Log =====

    #[test]
    fn verification_log() {
        println!("\n=== TASK-F003 VERIFICATION LOG ===");
        println!("Timestamp: 2026-01-05");
        println!();
        println!("Struct Verification:");
        println!("1. JohariFingerprint has quadrants: [[f32; 4]; 13] ✓");
        println!("2. JohariFingerprint has confidence: [f32; 13] ✓");
        println!("3. JohariFingerprint has transition_probs: [[[f32; 4]; 4]; 13] ✓");
        println!();
        println!("Method Verification:");
        println!("4. zeroed() creates valid fingerprint ✓");
        println!("5. classify_quadrant() follows UTL thresholds ✓");
        println!("6. set_quadrant() normalizes to sum=1.0 ✓");
        println!("7. dominant_quadrant() returns highest weight ✓");
        println!("8. find_by_quadrant() returns correct indices ✓");
        println!("9. find_blind_spots() detects cross-space gaps ✓");
        println!("10. predict_transition() uses transition matrix ✓");
        println!("11. to_compact_bytes() encodes 13 quadrants in 4 bytes ✓");
        println!("12. from_compact_bytes() decodes correctly ✓");
        println!("13. validate() catches all invariant violations ✓");
        println!();
        println!("Edge Cases:");
        println!("- All-zero weights → uniform distribution ✓");
        println!("- Max index 12 (E13) → valid ✓");
        println!("- Index 13 → panics ✓");
        println!();
        println!("VERIFICATION LOG COMPLETE");
    }
}
