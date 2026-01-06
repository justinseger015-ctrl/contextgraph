//! ImageProcessor tests for MultimodalModel.

use crate::error::EmbeddingError;
use crate::types::ImageFormat;

use super::super::ImageProcessor;
use super::create_minimal_png;

#[test]
fn test_image_processor_new() {
    let processor = ImageProcessor::new();
    assert_eq!(processor.target_size(), 224);
}

#[test]
fn test_image_processor_preprocess_png() {
    let processor = ImageProcessor::new();
    let png_bytes = create_minimal_png();
    let result = processor.preprocess(&png_bytes, ImageFormat::Png);
    assert!(result.is_ok());
    let tensor = result.unwrap();
    assert_eq!(tensor.len(), 224 * 224 * 3);
}

#[test]
fn test_image_processor_preprocess_empty_bytes_fails() {
    let processor = ImageProcessor::new();
    let result = processor.preprocess(&[], ImageFormat::Png);
    assert!(matches!(result, Err(EmbeddingError::EmptyInput)));
}

#[test]
fn test_image_processor_preprocess_invalid_bytes_fails() {
    let processor = ImageProcessor::new();
    let invalid_bytes = vec![0x00, 0x01, 0x02, 0x03];
    let result = processor.preprocess(&invalid_bytes, ImageFormat::Png);
    assert!(matches!(result, Err(EmbeddingError::InvalidImage { .. })));
}
