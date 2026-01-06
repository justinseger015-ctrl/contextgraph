//! Construction and initialization tests for HdcModel.

use super::*;

#[test]
fn test_new_with_valid_params() {
    let model = HdcModel::new(3, 0xDEAD_BEEF).unwrap();
    assert_eq!(model.ngram_size(), 3);
    assert_eq!(model.seed(), 0xDEAD_BEEF);
    assert!(model.is_initialized());
}

#[test]
fn test_new_with_ngram_size_1() {
    let model = HdcModel::new(1, 0).unwrap();
    assert_eq!(model.ngram_size(), 1);
}

#[test]
fn test_new_with_ngram_size_0_fails() {
    let result = HdcModel::new(0, 0);
    assert!(result.is_err());
    match result {
        Err(EmbeddingError::ConfigError { message }) => {
            assert!(message.contains("ngram_size"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_default_model() {
    let model = HdcModel::default_model();
    assert_eq!(model.ngram_size(), DEFAULT_NGRAM_SIZE);
    assert_eq!(model.seed(), DEFAULT_SEED);
    assert!(model.is_initialized());
}

#[test]
fn test_model_id() {
    let model = HdcModel::default_model();
    assert_eq!(model.model_id(), ModelId::Hdc);
}
