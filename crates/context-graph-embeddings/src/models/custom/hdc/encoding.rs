//! Text encoding for HDC.
//!
//! Provides text-to-hypervector encoding using character n-grams with
//! positional binding.

use super::operations::{bind, bundle, permute};
use super::types::{Hypervector, HDC_DIMENSION, HDC_PROJECTED_DIMENSION};
use bitvec::prelude::*;
use tracing::{debug, instrument, trace, warn};

/// Generates a random hypervector with approximately 50% bits set.
///
/// Uses a deterministic PRNG seeded with the combined seed and input hash.
///
/// # Arguments
/// * `seed` - Base seed for the model
/// * `key` - Unique identifier for this hypervector (hashed with seed)
#[must_use]
pub fn random_hypervector(seed: u64, key: u64) -> Hypervector {
    let mut hv = bitvec![u64, Lsb0; 0; HDC_DIMENSION];
    let mut state = seed.wrapping_add(key);

    for i in 0..HDC_DIMENSION {
        // LCG PRNG: state = state * 6364136223846793005 + 1
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Use high bits for better distribution
        if (state >> 63) == 1 {
            hv.set(i, true);
        }
    }

    trace!(
        key = key,
        popcount = hv.count_ones(),
        "Generated random hypervector"
    );
    hv
}

/// Generates a hypervector for a character.
///
/// Characters are hashed deterministically to produce consistent vectors.
#[must_use]
pub fn char_hypervector(seed: u64, c: char) -> Hypervector {
    random_hypervector(seed, c as u64)
}

/// Generates a position permutation vector.
///
/// Position vectors enable positional encoding via binding.
#[must_use]
pub fn position_hypervector(seed: u64, position: usize) -> Hypervector {
    // Use a different seed space for positions
    random_hypervector(seed, 0x8000_0000_0000_0000 | (position as u64))
}

/// Encodes text into a hypervector using character n-grams.
///
/// Algorithm:
/// 1. For each n-gram of characters
/// 2. Bind character vectors with position shifts
/// 3. Bundle all n-gram vectors
///
/// # Arguments
/// * `seed` - Model seed for deterministic generation
/// * `ngram_size` - Size of character n-grams
/// * `text` - Text to encode
///
/// # Returns
/// Hypervector encoding of the text
#[must_use]
#[instrument(skip(text), fields(text_len = text.len()))]
pub fn encode_text(seed: u64, ngram_size: usize, text: &str) -> Hypervector {
    let chars: Vec<char> = text.chars().collect();

    if chars.is_empty() {
        warn!("Encoding empty text");
        return bitvec![u64, Lsb0; 0; HDC_DIMENSION];
    }

    // For text shorter than ngram_size, use what we have
    let effective_ngram = ngram_size.min(chars.len());

    let mut ngram_vectors = Vec::new();

    for window_start in 0..=chars.len().saturating_sub(effective_ngram) {
        let mut ngram_hv = char_hypervector(seed, chars[window_start]);

        // Bind characters within n-gram with position shifts
        for (pos, &c) in chars[window_start..]
            .iter()
            .take(effective_ngram)
            .enumerate()
            .skip(1)
        {
            let char_hv = char_hypervector(seed, c);
            let shifted = permute(&char_hv, pos);
            ngram_hv = bind(&ngram_hv, &shifted);
        }

        // Bind with absolute position
        let pos_hv = position_hypervector(seed, window_start);
        ngram_hv = bind(&ngram_hv, &pos_hv);

        ngram_vectors.push(ngram_hv);
    }

    let result = bundle(&ngram_vectors);
    debug!(
        ngrams = ngram_vectors.len(),
        popcount = result.count_ones(),
        "Encoded text"
    );
    result
}

/// Projects a hypervector to the fusion dimension (1024D float).
///
/// Binary to float conversion: Uses weighted averaging of bit chunks.
/// Then L2 normalized.
#[must_use]
pub fn project_to_float(hv: &Hypervector) -> Vec<f32> {
    // Project from 10K bits to 1024 floats
    // Since 10000 doesn't divide evenly by 1024, we use overlapping windows
    let hv_len = hv.len();
    let mut result = Vec::with_capacity(HDC_PROJECTED_DIMENSION);

    for i in 0..HDC_PROJECTED_DIMENSION {
        // Map output index to input bit range
        let start = (i * hv_len) / HDC_PROJECTED_DIMENSION;
        let end = ((i + 1) * hv_len) / HDC_PROJECTED_DIMENSION;
        let chunk_size = end - start;

        if chunk_size == 0 {
            result.push(0.0);
            continue;
        }

        let ones: usize = (start..end)
            .map(|j| if j < hv_len && hv[j] { 1 } else { 0 })
            .sum();
        // Map [0, chunk_size] to [-1, +1]
        let value = (2.0 * ones as f32 / chunk_size as f32) - 1.0;
        result.push(value);
    }

    // L2 normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in &mut result {
            *v /= norm;
        }
    }

    result
}
