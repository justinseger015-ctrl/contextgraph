//! Tests for query access methods.

use crate::index::search_result::SearchResult;

#[test]
fn test_query_results_basic() {
    let result = SearchResult::new(
        vec![1, 2, 3, 4, 5, 6],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        3,
        2,
    );

    let q0: Vec<_> = result.query_results(0).collect();
    assert_eq!(q0, vec![(1, 0.1), (2, 0.2), (3, 0.3)]);

    let q1: Vec<_> = result.query_results(1).collect();
    assert_eq!(q1, vec![(4, 0.4), (5, 0.5), (6, 0.6)]);
}

// ========== Sentinel Filtering Tests ==========

#[test]
fn test_filter_sentinel_ids() {
    let result = SearchResult::new(
        vec![1, -1, 3, -1, -1, -1],
        vec![0.1, 0.0, 0.3, 0.0, 0.0, 0.0],
        3,
        2,
    );

    let q0: Vec<_> = result.query_results(0).collect();
    assert_eq!(q0, vec![(1, 0.1), (3, 0.3)]);

    let q1: Vec<_> = result.query_results(1).collect();
    assert!(q1.is_empty(), "All -1 sentinels should be filtered");
}

#[test]
fn test_all_sentinels_returns_empty() {
    let result = SearchResult::new(vec![-1, -1, -1], vec![0.0, 0.0, 0.0], 3, 1);

    assert!(result.query_results(0).next().is_none());
    assert!(!result.has_results(0));
    assert!(result.is_empty());
}

#[test]
fn test_partial_sentinels() {
    let result = SearchResult::new(vec![100, -1, -1], vec![0.5, 0.0, 0.0], 3, 1);

    let q: Vec<_> = result.query_results(0).collect();
    assert_eq!(q, vec![(100, 0.5)]);
    assert_eq!(result.query_results(0).count(), 1);
    assert!(result.has_results(0));
}

// ========== Count and Check Methods ==========

#[test]
fn test_valid_results_count() {
    let result = SearchResult::new(vec![1, -1, 3], vec![0.1, 0.0, 0.3], 3, 1);

    assert_eq!(result.query_results(0).count(), 2);
    assert_eq!(result.total_valid_results(), 2);
}

#[test]
fn test_has_results() {
    let result = SearchResult::new(
        vec![1, 2, 3, -1, -1, -1],
        vec![0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        3,
        2,
    );

    assert!(result.has_results(0));
    assert!(!result.has_results(1));
}

// ========== Top Result Tests ==========

#[test]
fn test_top_result_exists() {
    let result = SearchResult::new(vec![42, 43, 44], vec![0.5, 0.6, 0.7], 3, 1);

    let top = result.top_result(0);
    assert_eq!(top, Some((42, 0.5)));
}

#[test]
fn test_top_result_skips_sentinels() {
    let result = SearchResult::new(
        vec![-1, 42, 43], // First is sentinel
        vec![0.0, 0.5, 0.6],
        3,
        1,
    );

    let top = result.top_result(0);
    assert_eq!(top, Some((42, 0.5)));
}

#[test]
fn test_top_result_none_when_all_sentinels() {
    let result = SearchResult::new(vec![-1, -1, -1], vec![0.0, 0.0, 0.0], 3, 1);

    assert!(result.top_result(0).is_none());
}

// ========== All Results Iterator ==========

#[test]
fn test_all_results_iterator() {
    let result = SearchResult::new(vec![1, 2, 3, 4], vec![0.1, 0.2, 0.3, 0.4], 2, 2);

    let all: Vec<_> = result.all_results().collect();
    assert_eq!(
        all,
        vec![
            (0, 1, 0.1),
            (0, 2, 0.2), // Query 0
            (1, 3, 0.3),
            (1, 4, 0.4), // Query 1
        ]
    );
}

#[test]
fn test_all_results_filters_sentinels() {
    let result = SearchResult::new(vec![1, -1, 3, -1], vec![0.1, 0.0, 0.3, 0.0], 2, 2);

    let all: Vec<_> = result.all_results().collect();
    assert_eq!(
        all,
        vec![
            (0, 1, 0.1), // Query 0: only ID 1
            (1, 3, 0.3), // Query 1: only ID 3
        ]
    );
}

// ========== Panic Tests ==========

#[test]
#[should_panic(expected = "query_idx (1) >= num_queries (1)")]
fn test_query_idx_out_of_bounds() {
    let result = SearchResult::new(vec![1, 2], vec![0.1, 0.2], 2, 1);
    let _ = result.query_results(1).collect::<Vec<_>>();
}

#[test]
#[should_panic(expected = "query_idx (5) >= num_queries (2)")]
fn test_query_idx_way_out_of_bounds() {
    let result = SearchResult::new(vec![1, 2, 3, 4], vec![0.1, 0.2, 0.3, 0.4], 2, 2);
    let _ = result.query_results(5).collect::<Vec<_>>();
}
