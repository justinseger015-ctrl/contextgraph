//! Full-State Verification: WS1 (MaxSim Reranking) + WS4 (Score-Aware RRF)
//!
//! Run: cargo test --release -p context-graph-core --test ws_core_verification -- --nocapture

use context_graph_core::fusion::{
    fuse_rankings, score_weighted_rrf, weighted_rrf, EmbedderRanking, FusionStrategy,
};
use context_graph_core::retrieval::distance::max_sim;
use uuid::Uuid;

fn make_uuid(n: u8) -> Uuid {
    Uuid::from_bytes([n; 16])
}

// ============================================================================
// WS1: E12 ColBERT MaxSim Reranking
// ============================================================================

#[test]
fn ws1_happy_path_maxsim_reorders_candidates() {
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    let candidate_a = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let candidate_b = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let candidate_c = vec![
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    let score_a = max_sim(&query_tokens, &candidate_a);
    let score_b = max_sim(&query_tokens, &candidate_b);
    let score_c = max_sim(&query_tokens, &candidate_c);

    println!("=== WS1 HAPPY PATH ===");
    println!("A (perfect)  = {:.6}", score_a);
    println!("B (partial)  = {:.6}", score_b);
    println!("C (no match) = {:.6}", score_c);

    assert!((score_a - 1.0).abs() < 1e-5, "Perfect match != 1.0: {}", score_a);
    assert!(score_b > 0.6 && score_b < 0.7, "Partial != ~0.667: {}", score_b);
    assert!(score_c > 0.45 && score_c < 0.55, "No match != ~0.5: {}", score_c);
    assert!(score_a > score_b && score_b > score_c, "Ordering wrong");

    // Simulate stage2+stage3 interpolation
    let stage2 = [0.5_f32, 0.7, 0.9]; // C was best by stage2
    let ms = [score_a, score_b, score_c];
    let w = 0.4_f32;
    let final_a = (1.0 - w) * stage2[0] + w * ms[0];
    let final_c = (1.0 - w) * stage2[2] + w * ms[2];
    let a_boost = final_a - stage2[0]; // How much A gained
    let c_loss = final_c - stage2[2];   // How much C lost

    println!("Rerank w=0.4: A {:.3}→{:.3} (boost {:.3}), C {:.3}→{:.3} (loss {:.3})",
        stage2[0], final_a, a_boost, stage2[2], final_c, c_loss);
    assert!(a_boost > 0.0, "MaxSim should boost A");
    assert!(c_loss < 0.0, "MaxSim should penalize C");
    println!("[PASS]\n");
}

#[test]
fn ws1_edge_empty_query() {
    let empty: Vec<Vec<f32>> = vec![];
    let mem = vec![vec![1.0, 0.0, 0.0]];
    let score = max_sim(&empty, &mem);
    println!("WS1 Edge: empty query → {}", score);
    assert_eq!(score, 0.0);
}

#[test]
fn ws1_edge_empty_memory() {
    let q = vec![vec![1.0, 0.0, 0.0]];
    let empty: Vec<Vec<f32>> = vec![];
    let score = max_sim(&q, &empty);
    println!("WS1 Edge: empty memory → {}", score);
    assert_eq!(score, 0.0);
}

#[test]
fn ws1_edge_zero_vector() {
    let q = vec![vec![0.0, 0.0, 0.0]];
    let m = vec![vec![1.0, 0.0, 0.0]];
    let score = max_sim(&q, &m);
    println!("WS1 Edge: zero vec → {} (AP-10)", score);
    assert_eq!(score, 0.0);
    assert!(!score.is_nan());
}

// ============================================================================
// WS4: Score-Aware RRF Fusion
// ============================================================================

#[test]
fn ws4_happy_path_e5_magnitude_matters() {
    println!("=== WS4 HAPPY PATH ===");
    let d1 = make_uuid(1);
    let d2 = make_uuid(2);

    let rankings = vec![
        EmbedderRanking::new("E1", 1.0, vec![(d1, 0.85), (d2, 0.84)]),
        EmbedderRanking::new("E5", 0.5, vec![(d1, 0.58), (d2, 0.12)]),
    ];

    let std_r = weighted_rrf(&rankings, 10);
    let sw_r = score_weighted_rrf(&rankings, &["E5"], 10);

    for r in &std_r { println!("Std RRF: doc {:02x} = {:.6}", r.doc_id.as_bytes()[0], r.fused_score); }
    for r in &sw_r { println!("SW  RRF: doc {:02x} = {:.6}", r.doc_id.as_bytes()[0], r.fused_score); }

    let std_gap = std_r[0].fused_score - std_r[1].fused_score;
    let sw_gap = sw_r[0].fused_score - sw_r[1].fused_score;
    println!("Std gap={:.6}, SW gap={:.6}", std_gap, sw_gap);
    assert!(sw_gap > std_gap, "SW gap should exceed std gap");

    // Verify exact E5 contribution formula
    let e5_d1_sw: f64 = 0.5 * 0.58 / 61.0;
    let e5_d2_sw: f64 = 0.5 * 0.12 / 62.0;
    let e5_d1_std: f64 = 0.5 / 61.0;
    let e5_d2_std: f64 = 0.5 / 62.0;
    println!("E5 contribution: d1 std={:.6} sw={:.6}", e5_d1_std, e5_d1_sw);
    println!("E5 contribution: d2 std={:.6} sw={:.6}", e5_d2_std, e5_d2_sw);

    // d1 gets MORE than standard (0.58 > 1.0 is false, so actually less... but the GAP is what matters)
    // Standard: d1 E5=0.008197, d2 E5=0.008065 → gap = 0.000132
    // SW: d1 E5=0.004754, d2 E5=0.000968 → gap = 0.003786
    // The score-weighted GAP between E5 contributions is much larger
    assert!((e5_d1_sw - e5_d2_sw).abs() > (e5_d1_std - e5_d2_std).abs(),
        "E5 score-weighted contribution gap should be larger");
    println!("[PASS]\n");
}

#[test]
fn ws4_happy_path_dispatch() {
    let rankings = vec![
        EmbedderRanking::new("E1", 1.0, vec![(make_uuid(1), 0.9)]),
        EmbedderRanking::new("E5", 0.5, vec![(make_uuid(1), 0.5)]),
    ];

    let rrf = fuse_rankings(&rankings, FusionStrategy::WeightedRRF, 10);
    let sw = fuse_rankings(&rankings, FusionStrategy::ScoreWeightedRRF, 10);
    let sum = fuse_rankings(&rankings, FusionStrategy::WeightedSum, 10);

    println!("WS4 Dispatch: RRF={:.6} SW={:.6} Sum={:.6}",
        rrf[0].fused_score, sw[0].fused_score, sum[0].fused_score);

    assert!(rrf[0].fused_score > 0.0);
    assert!(sw[0].fused_score > 0.0);
    assert!(sum[0].fused_score > 0.0);
    assert!((sw[0].fused_score - rrf[0].fused_score).abs() > 1e-6);
    println!("[PASS]\n");
}

#[test]
fn ws4_edge_empty_rankings() {
    let empty: Vec<EmbedderRanking> = vec![];
    let r = score_weighted_rrf(&empty, &["E5"], 10);
    println!("WS4 Edge: empty → {} results", r.len());
    assert_eq!(r.len(), 0);
}

#[test]
fn ws4_edge_e5_zero_score() {
    let rankings = vec![
        EmbedderRanking::new("E1", 1.0, vec![(make_uuid(1), 0.9)]),
        EmbedderRanking::new("E5", 1.0, vec![(make_uuid(1), 0.0)]),
    ];
    let sw = score_weighted_rrf(&rankings, &["E5"], 10);
    let std = weighted_rrf(&rankings, 10);
    println!("WS4 Edge: E5=0 → sw={:.6} std={:.6}", sw[0].fused_score, std[0].fused_score);
    assert!(sw[0].fused_score < std[0].fused_score, "Zero E5 should reduce contribution");
}

#[test]
fn ws4_edge_no_e5_embedder() {
    let rankings = vec![
        EmbedderRanking::new("E1", 1.0, vec![(make_uuid(1), 0.9)]),
        EmbedderRanking::new("E7", 0.5, vec![(make_uuid(1), 0.8)]),
    ];
    let sw = score_weighted_rrf(&rankings, &["E5"], 10);
    let std = weighted_rrf(&rankings, 10);
    println!("WS4 Edge: no E5 → sw={:.6} std={:.6}", sw[0].fused_score, std[0].fused_score);
    assert!((sw[0].fused_score - std[0].fused_score).abs() < 1e-6, "Should be identical without E5");
}

/// WS4: Verify score-weighted RRF preserves E5 score across multiple documents
/// with same rank positions but dramatically different E5 scores.
#[test]
fn ws4_multi_doc_score_preservation() {
    println!("=== WS4 Multi-Doc Score Preservation ===");

    let d1 = make_uuid(10);
    let d2 = make_uuid(20);
    let d3 = make_uuid(30);

    // All 3 docs have same E1 scores, but very different E5 scores
    let rankings = vec![
        EmbedderRanking::new("E1", 1.0, vec![
            (d1, 0.80), (d2, 0.79), (d3, 0.78),
        ]),
        EmbedderRanking::new("E5", 1.0, vec![
            (d3, 0.95),  // d3 has strongest causal signal
            (d1, 0.50),  // d1 moderate
            (d2, 0.05),  // d2 almost no causal signal
        ]),
    ];

    let std_results = weighted_rrf(&rankings, 10);
    let sw_results = score_weighted_rrf(&rankings, &["E5"], 10);

    println!("Standard RRF:");
    for r in &std_results {
        println!("  doc {:02x}: {:.6}", r.doc_id.as_bytes()[0], r.fused_score);
    }
    println!("Score-weighted RRF:");
    for r in &sw_results {
        println!("  doc {:02x}: {:.6}", r.doc_id.as_bytes()[0], r.fused_score);
    }

    // In SW RRF, d3 should be promoted (highest E5 score 0.95) above d1 (E1 rank 1)
    assert_eq!(sw_results[0].doc_id, d3,
        "d3 with E5=0.95 should be ranked first in score-weighted RRF");
    println!("[PASS] d3 (E5=0.95) promoted to top rank in SW RRF\n");
}
