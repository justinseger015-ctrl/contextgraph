//! Periodicity and Fourier property tests for TemporalPeriodicModel.
//!
//! These tests verify the key differentiating behavior of TemporalPeriodic:
//! its ability to encode cyclical time patterns.

use crate::models::custom::temporal_periodic::{TemporalPeriodicModel, FEATURES_PER_PERIOD};
use crate::traits::EmbeddingModel;
use crate::types::ModelInput;

/// Helper function to compute cosine similarity between two feature vectors.
/// Normalizes both vectors before computing dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot_product / (norm_a * norm_b)
}

#[tokio::test]
async fn test_deterministic_with_same_timestamp() {
    let model = TemporalPeriodicModel::new();

    let timestamp = "timestamp:2024-01-15T10:30:00Z";
    let input1 = ModelInput::text_with_instruction("content", timestamp).expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", timestamp).expect("Failed to create");

    let embedding1 = model.embed(&input1).await.expect("First embed");
    let embedding2 = model.embed(&input2).await.expect("Second embed");

    assert_eq!(
        embedding1.vector, embedding2.vector,
        "Same timestamp must produce identical embeddings"
    );
}

#[tokio::test]
async fn test_different_timestamps_different_embeddings() {
    let model = TemporalPeriodicModel::new();

    let input1 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T14:30:00Z")
        .expect("Failed to create");

    let embedding1 = model.embed(&input1).await.expect("First embed");
    let embedding2 = model.embed(&input2).await.expect("Second embed");

    assert_ne!(
        embedding1.vector, embedding2.vector,
        "Different timestamps must produce different embeddings"
    );
}

#[tokio::test]
async fn test_same_time_of_day_similar_hour_features() {
    let model = TemporalPeriodicModel::new();

    // Same time on different days - 10:30 AM on consecutive days
    let input1 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", "timestamp:2024-01-16T10:30:00Z")
        .expect("Failed to create");

    let emb1 = model.embed(&input1).await.expect("First embed");
    let emb2 = model.embed(&input2).await.expect("Second embed");

    // Hour features (first 102 values) should be identical for same time-of-day
    let hour_features_1: &[f32] = &emb1.vector[0..FEATURES_PER_PERIOD];
    let hour_features_2: &[f32] = &emb2.vector[0..FEATURES_PER_PERIOD];

    // Calculate true cosine similarity (normalize subsets first)
    let cosine_sim = cosine_similarity(hour_features_1, hour_features_2);

    println!("Hour features cosine similarity: {}", cosine_sim);

    // Same time of day should have very high similarity in hour features
    assert!(
        cosine_sim > 0.99,
        "Same time-of-day should have near-identical hour features, got {}",
        cosine_sim
    );
}

#[tokio::test]
async fn test_same_day_of_week_similar_week_features() {
    let model = TemporalPeriodicModel::new();

    // Same weekday (Monday) on different weeks
    let input1 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z") // Monday
        .expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", "timestamp:2024-01-22T10:30:00Z") // Monday next week
        .expect("Failed to create");

    let emb1 = model.embed(&input1).await.expect("First embed");
    let emb2 = model.embed(&input2).await.expect("Second embed");

    // Week features (positions 204-306, third period block)
    let week_start = FEATURES_PER_PERIOD * 2; // Skip hour and day
    let week_end = week_start + FEATURES_PER_PERIOD;
    let week_features_1: &[f32] = &emb1.vector[week_start..week_end];
    let week_features_2: &[f32] = &emb2.vector[week_start..week_end];

    let cosine_sim = cosine_similarity(week_features_1, week_features_2);

    println!("Week features cosine similarity: {}", cosine_sim);

    assert!(
        cosine_sim > 0.99,
        "Same day-of-week should have near-identical week features, got {}",
        cosine_sim
    );
}

#[tokio::test]
async fn test_different_time_of_day_different_day_features() {
    let model = TemporalPeriodicModel::new();

    // Different times on same day - 6:00 AM vs 6:00 PM (12 hours apart = half daily cycle)
    let input1 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T06:00:00Z")
        .expect("Failed to create");
    let input2 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T18:00:00Z")
        .expect("Failed to create");

    let emb1 = model.embed(&input1).await.expect("First embed");
    let emb2 = model.embed(&input2).await.expect("Second embed");

    // Day features (period = DAY = 86400 seconds) should be different
    // 6 AM vs 6 PM = 12 hours apart = half of daily cycle
    // The first period is HOUR (3600s), second is DAY (86400s)
    let day_start = FEATURES_PER_PERIOD; // Skip hour features
    let day_end = day_start + FEATURES_PER_PERIOD;
    let day_features_1: &[f32] = &emb1.vector[day_start..day_end];
    let day_features_2: &[f32] = &emb2.vector[day_start..day_end];

    let cosine_sim = cosine_similarity(day_features_1, day_features_2);

    println!("Day features cosine sim (6AM vs 6PM): {}", cosine_sim);

    // 6 AM and 6 PM are opposite sides of the daily cycle (half cycle apart)
    // 12 hours = 43200 seconds = 0.5 of 86400 seconds
    // With multiple harmonics, phase difference of pi gives complex behavior:
    // - Odd harmonics: sin(n*(theta+pi)) = -sin(n*theta) and cos(n*(theta+pi)) = -cos(n*theta)
    // - Even harmonics: sin(n*(theta+pi)) = sin(n*theta) and cos(n*(theta+pi)) = cos(n*theta)
    // This results in near-zero cosine similarity (orthogonal), not exactly -1
    // Key assertion: they should NOT be highly similar (not > 0.5)
    assert!(
        cosine_sim < 0.5,
        "6 AM and 6 PM should not be similar (half daily cycle), got {}",
        cosine_sim
    );
}

#[tokio::test]
async fn test_annual_cycle_same_date_different_years() {
    let model = TemporalPeriodicModel::new();

    // Same date (Jan 15) in different years
    let input1 = ModelInput::text_with_instruction("content", "timestamp:2023-01-15T10:30:00Z")
        .expect("year1");
    let input2 = ModelInput::text_with_instruction("content", "timestamp:2024-01-15T10:30:00Z")
        .expect("year2");

    let emb1 = model.embed(&input1).await.expect("Embed year1");
    let emb2 = model.embed(&input2).await.expect("Embed year2");

    // Year features (positions 408-510, fifth period block)
    let year_start = FEATURES_PER_PERIOD * 4;
    let year_end = year_start + FEATURES_PER_PERIOD;
    let year_features_1: &[f32] = &emb1.vector[year_start..year_end];
    let year_features_2: &[f32] = &emb2.vector[year_start..year_end];

    let cosine_sim = cosine_similarity(year_features_1, year_features_2);

    println!(
        "Year features cosine similarity (same date different years): {}",
        cosine_sim
    );

    // Same date in different years should have similar year-cycle features
    assert!(
        cosine_sim > 0.99,
        "Same date different years should have similar year features, got {}",
        cosine_sim
    );
}
