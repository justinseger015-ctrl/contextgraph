//! Edge cases and special scenarios tests.

use super::*;

#[test]
fn test_single_character_text() {
    let model = HdcModel::default_model();
    let hv = model.encode_text("x");
    assert_eq!(hv.len(), HDC_DIMENSION);
    assert!(hv.count_ones() > 0);
}

#[test]
fn test_very_long_text() {
    let model = HdcModel::default_model();
    let long_text = "abc".repeat(10_000);
    let hv = model.encode_text(&long_text);
    assert_eq!(hv.len(), HDC_DIMENSION);
}

#[test]
fn test_unicode_text() {
    let model = HdcModel::default_model();
    let hv1 = model.encode_text("こんにちは世界");
    let hv2 = model.encode_text("Hello World");
    assert_eq!(hv1.len(), HDC_DIMENSION);
    assert_eq!(hv2.len(), HDC_DIMENSION);
    assert_ne!(hv1, hv2);
}

#[test]
fn test_emoji_text() {
    let model = HdcModel::default_model();
    let hv = model.encode_text("Hello World !");
    assert_eq!(hv.len(), HDC_DIMENSION);
    assert!(hv.count_ones() > 0);
}

#[tokio::test]
async fn test_text_with_instruction() {
    let model = HdcModel::default_model();
    let input = ModelInput::text_with_instruction("Query text", "query:").unwrap();
    let result = model.embed(&input).await;
    assert!(result.is_ok());
}

// ========================================================================
// SOURCE OF TRUTH VERIFICATION
// ========================================================================

#[test]
fn test_hdc_dimension_matches_model_id() {
    assert_eq!(ModelId::Hdc.dimension(), HDC_DIMENSION);
}

#[test]
fn test_hdc_projected_dimension_matches_model_id() {
    assert_eq!(ModelId::Hdc.projected_dimension(), HDC_PROJECTED_DIMENSION);
}

// ========================================================================
// THREAD SAFETY TESTS
// ========================================================================

#[test]
fn test_hdc_model_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<HdcModel>();
}

#[test]
fn test_hdc_model_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<HdcModel>();
}

// ========================================================================
// CONCURRENT USAGE TEST
// ========================================================================

#[tokio::test]
async fn test_concurrent_embedding() {
    use std::sync::Arc;

    let model = Arc::new(HdcModel::default_model());

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let m = Arc::clone(&model);
            tokio::spawn(async move {
                let input = ModelInput::text(format!("Concurrent test {}", i)).unwrap();
                m.embed(&input).await
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.model_id, ModelId::Hdc);
        assert_eq!(embedding.dimension(), HDC_PROJECTED_DIMENSION);
    }
}
