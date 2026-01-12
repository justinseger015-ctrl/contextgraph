//! Tests for FeedbackLearner module.
//!
//! Tests are split across multiple files to stay under the 500 line limit:
//! - config_tests: FeedbackLearnerConfig tests
//! - types_tests: FeedbackType, FeedbackEvent, LearningResult tests  
//! - learner_tests: FeedbackLearner core tests
//! - gradient_tests: Gradient computation and learning cycle tests

mod config_tests;
mod gradient_tests;
mod learner_tests;
mod types_tests;
