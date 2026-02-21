//! Secondary index operation tests.

use chrono::{Duration, Utc};

use super::tests_node::{create_node_with_tags, create_temp_db, create_valid_test_node};

#[test]
fn test_get_nodes_by_tag_finds_nodes() {
    let (_tmp, db) = create_temp_db();

    let node1 = create_node_with_tags(vec!["important", "work"]);
    db.store_node(&node1).unwrap();
    let node2 = create_node_with_tags(vec!["important", "personal"]);
    db.store_node(&node2).unwrap();
    let node3 = create_node_with_tags(vec!["personal"]);
    db.store_node(&node3).unwrap();

    let important = db.get_nodes_by_tag("important", None, 0).unwrap();
    let personal = db.get_nodes_by_tag("personal", None, 0).unwrap();
    let work = db.get_nodes_by_tag("work", None, 0).unwrap();

    assert_eq!(important.len(), 2);
    assert_eq!(personal.len(), 2);
    assert_eq!(work.len(), 1);
}

#[test]
fn test_get_nodes_by_tag_with_pagination() {
    let (_tmp, db) = create_temp_db();

    for _ in 0..10 {
        let node = create_node_with_tags(vec!["paginated"]);
        db.store_node(&node).unwrap();
    }

    let page1 = db.get_nodes_by_tag("paginated", Some(3), 0).unwrap();
    let page2 = db.get_nodes_by_tag("paginated", Some(3), 3).unwrap();

    assert_eq!(page1.len(), 3);
    assert_eq!(page2.len(), 3);
    for id in &page1 {
        assert!(!page2.contains(id), "Pages should not overlap");
    }
}

#[test]
fn test_get_nodes_by_source_finds_nodes() {
    let (_tmp, db) = create_temp_db();

    let mut node1 = create_valid_test_node();
    node1.metadata.source = Some("api-gateway".to_string());
    db.store_node(&node1).unwrap();

    let mut node2 = create_valid_test_node();
    node2.metadata.source = Some("api-gateway".to_string());
    db.store_node(&node2).unwrap();

    let api_nodes = db.get_nodes_by_source("api-gateway", None, 0).unwrap();
    assert_eq!(api_nodes.len(), 2);
}

#[test]
fn test_get_nodes_in_time_range_finds_nodes() {
    let (_tmp, db) = create_temp_db();

    let node1 = create_valid_test_node();
    let created_at = node1.created_at;
    db.store_node(&node1).unwrap();
    let node2 = create_valid_test_node();
    db.store_node(&node2).unwrap();

    let start = created_at - Duration::seconds(1);
    let end = Utc::now() + Duration::seconds(1);

    let result = db.get_nodes_in_time_range(start, end, None, 0).unwrap();
    assert_eq!(result.len(), 2);
}

#[test]
fn test_get_nodes_in_time_range_respects_boundaries() {
    let (_tmp, db) = create_temp_db();

    let node = create_valid_test_node();
    let node_time = node.created_at;
    db.store_node(&node).unwrap();

    let before = db.get_nodes_in_time_range(
        node_time - Duration::hours(2),
        node_time - Duration::hours(1),
        None, 0,
    ).unwrap();
    assert!(before.is_empty());

    let including = db.get_nodes_in_time_range(
        node_time - Duration::seconds(1),
        node_time + Duration::seconds(1),
        None, 0,
    ).unwrap();
    assert_eq!(including.len(), 1);
}

#[test]
fn test_get_nodes_by_tag_similar_tags() {
    let (_tmp, db) = create_temp_db();

    let node1 = create_node_with_tags(vec!["test"]);
    db.store_node(&node1).unwrap();
    let node2 = create_node_with_tags(vec!["testing"]);
    db.store_node(&node2).unwrap();

    assert_eq!(db.get_nodes_by_tag("test", None, 0).unwrap().len(), 1);
    assert_eq!(db.get_nodes_by_tag("testing", None, 0).unwrap().len(), 1);
}

#[test]
fn evidence_index_consistency() {
    let (_tmp, db) = create_temp_db();

    let mut node = create_node_with_tags(vec!["evidence", "verification"]);
    node.metadata.source = Some("evidence-source".to_string());
    let node_id = node.id;
    let created_at = node.created_at;
    db.store_node(&node).unwrap();

    assert!(db.get_nodes_by_tag("evidence", None, 0).unwrap().contains(&node_id));
    assert!(db.get_nodes_by_tag("verification", None, 0).unwrap().contains(&node_id));
    assert!(db.get_nodes_by_source("evidence-source", None, 0).unwrap().contains(&node_id));

    let temporal = db.get_nodes_in_time_range(
        created_at - Duration::seconds(1),
        created_at + Duration::seconds(1),
        None, 0,
    ).unwrap();
    assert!(temporal.contains(&node_id));
}

#[test]
fn edge_case_tag_with_special_chars() {
    let (_tmp, db) = create_temp_db();

    for tag in &["tag:with:colons", "tag/with/slashes", "tag-with-dashes"] {
        let node = create_node_with_tags(vec![tag]);
        db.store_node(&node).unwrap();
        assert_eq!(db.get_nodes_by_tag(tag, None, 0).unwrap().len(), 1);
    }
}
