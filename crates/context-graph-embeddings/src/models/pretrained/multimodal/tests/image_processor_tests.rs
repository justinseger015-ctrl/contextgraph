//! ImageProcessor tests for MultimodalModel.

use super::super::ImageProcessor;

#[test]
fn test_image_processor_new() {
    let processor = ImageProcessor::new();
    assert_eq!(processor.target_size(), 224);
}
