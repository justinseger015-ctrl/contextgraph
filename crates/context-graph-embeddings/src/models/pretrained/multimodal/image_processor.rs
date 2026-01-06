//! Image processor for CLIP-specific preprocessing.
//!
//! Handles the image preprocessing pipeline required by CLIP:
//! - Decode image bytes (PNG, JPEG, WebP, GIF)
//! - Resize to 224x224 with bilinear interpolation
//! - Convert to RGB (handle grayscale, RGBA)
//! - Normalize with CLIP mean/std values

use image::{DynamicImage, GenericImageView, ImageFormat as ImgFormat};

use crate::error::{EmbeddingError, EmbeddingResult};
use crate::types::ImageFormat;

use super::constants::{CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD};

/// Image processor for CLIP-specific preprocessing.
///
/// Handles the image preprocessing pipeline required by CLIP:
/// 1. Decode image bytes (PNG, JPEG, WebP, GIF)
/// 2. Resize to 224x224 with bilinear interpolation
/// 3. Convert to RGB (handle grayscale, RGBA)
/// 4. Normalize with CLIP mean/std values
///
/// # Example
///
/// ```rust,no_run
/// use context_graph_embeddings::models::ImageProcessor;
/// use context_graph_embeddings::types::ImageFormat;
/// use context_graph_embeddings::error::EmbeddingResult;
///
/// fn example() -> EmbeddingResult<()> {
///     let processor = ImageProcessor::new();
///
///     // Process image bytes
///     let png_bytes = std::fs::read("image.png").map_err(|e| {
///         context_graph_embeddings::error::EmbeddingError::InvalidImage {
///             reason: e.to_string(),
///         }
///     })?;
///     let tensor = processor.preprocess(&png_bytes, ImageFormat::Png)?;
///     // tensor is 224x224x3 normalized float tensor
///     Ok(())
/// }
/// ```
pub struct ImageProcessor {
    /// Target image size (224 for CLIP).
    target_size: u32,
    /// RGB mean values for normalization.
    mean: [f32; 3],
    /// RGB std values for normalization.
    std: [f32; 3],
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageProcessor {
    /// Create a new ImageProcessor with CLIP-specific parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_size: CLIP_IMAGE_SIZE,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Preprocess image bytes for CLIP embedding.
    ///
    /// # Arguments
    /// * `bytes` - Raw encoded image bytes (PNG, JPEG, etc.)
    /// * `format` - Image format for proper decoding
    ///
    /// # Returns
    /// Flattened normalized RGB tensor as Vec<f32> (224*224*3 = 150528 values).
    ///
    /// # Errors
    /// - `EmbeddingError::InvalidImage` if decoding fails
    /// - `EmbeddingError::EmptyInput` if bytes is empty
    pub fn preprocess(&self, bytes: &[u8], format: ImageFormat) -> EmbeddingResult<Vec<f32>> {
        if bytes.is_empty() {
            return Err(EmbeddingError::EmptyInput);
        }

        // Decode image
        let img = self.decode_image(bytes, format)?;

        // Resize to target size (224x224)
        let resized = img.resize_exact(
            self.target_size,
            self.target_size,
            image::imageops::FilterType::Triangle, // Bilinear interpolation
        );

        // Convert to RGB8
        let rgb = resized.to_rgb8();

        // Normalize and flatten to tensor
        let mut tensor = Vec::with_capacity((self.target_size * self.target_size * 3) as usize);

        for pixel in rgb.pixels() {
            // Normalize each channel: (value/255 - mean) / std
            let r = (pixel[0] as f32 / 255.0 - self.mean[0]) / self.std[0];
            let g = (pixel[1] as f32 / 255.0 - self.mean[1]) / self.std[1];
            let b = (pixel[2] as f32 / 255.0 - self.mean[2]) / self.std[2];

            tensor.push(r);
            tensor.push(g);
            tensor.push(b);
        }

        Ok(tensor)
    }

    /// Decode image bytes into DynamicImage.
    fn decode_image(&self, bytes: &[u8], format: ImageFormat) -> EmbeddingResult<DynamicImage> {
        let img_format = match format {
            ImageFormat::Png => ImgFormat::Png,
            ImageFormat::Jpeg => ImgFormat::Jpeg,
            ImageFormat::WebP => ImgFormat::WebP,
            ImageFormat::Gif => ImgFormat::Gif,
        };

        image::load_from_memory_with_format(bytes, img_format).map_err(|e| {
            tracing::error!("Failed to decode image: {}", e);
            EmbeddingError::InvalidImage {
                reason: format!("Failed to decode {} image: {}", format, e),
            }
        })
    }

    /// Get the target image size.
    #[must_use]
    pub const fn target_size(&self) -> u32 {
        self.target_size
    }

    /// Validate image dimensions after decoding.
    #[allow(dead_code)]
    fn validate_dimensions(&self, img: &DynamicImage) -> EmbeddingResult<()> {
        let (width, height) = img.dimensions();
        if width == 0 || height == 0 {
            return Err(EmbeddingError::InvalidImage {
                reason: "Image has zero dimensions".to_string(),
            });
        }
        Ok(())
    }
}
