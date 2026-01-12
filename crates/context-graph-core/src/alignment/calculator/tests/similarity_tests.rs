//! Cosine similarity tests.

use crate::alignment::calculator::similarity;
use crate::alignment::calculator::DefaultAlignmentCalculator;
use crate::purpose::GoalLevel;

#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: identical vectors = 1.0");
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!(sim.abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: orthogonal vectors = 0.0");
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 0.001);
    println!("[VERIFIED] cosine_similarity: opposite vectors = -1.0");
}

#[test]
fn test_cosine_similarity_mismatched_dims() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let sim = similarity::cosine_similarity(&a, &b);
    assert_eq!(sim, 0.0);
    println!("[VERIFIED] cosine_similarity: mismatched dims = 0.0");
}

#[test]
fn test_propagation_weights() {
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::NorthStar),
        1.0
    );
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Strategic),
        0.7
    );
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Tactical),
        0.4
    );
    assert_eq!(
        DefaultAlignmentCalculator::get_propagation_weight(GoalLevel::Immediate),
        0.2
    );
    println!("[VERIFIED] Propagation weights match TASK-L003 spec");
}
