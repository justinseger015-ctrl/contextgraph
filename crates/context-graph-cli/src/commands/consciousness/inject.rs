//! consciousness inject-context CLI command
//!
//! TASK-SESSION-15: Injects session context into LLM requests.
//! NO BACKWARDS COMPATIBILITY - FAIL FAST WITH ROBUST LOGGING.
//!
//! # Purpose
//!
//! Provides consciousness context for LLM prompt injection. Reads from cache
//! if warm, otherwise loads from RocksDB. Classifies Johari quadrant to
//! recommend appropriate action (direct recall, discovery, introspection).
//!
//! # Output Formats
//!
//! - `compact`: Single line, ~40 tokens max (for token-constrained prompts)
//! - `standard`: Multi-line with labels, ~100 tokens (default)
//! - `verbose`: Full diagnostic output with all fields
//!
//! # Johari Classification (per constitution.yaml)
//!
//! Using default thresholds: ΔS_threshold=0.5, ΔC_threshold=0.5
//! - **Open**: ΔS < 0.5 AND ΔC > 0.5 → Direct recall (get_node)
//! - **Blind**: ΔS > 0.5 AND ΔC < 0.5 → Discovery (epistemic_action/dream)
//! - **Hidden**: ΔS < 0.5 AND ΔC < 0.5 → Private (get_neighborhood)
//! - **Unknown**: ΔS > 0.5 AND ΔC > 0.5 → Frontier (explore)
//!
//! # Performance Target
//! - Latency: <1s total
//! - Token budget: ~40 tokens for compact, ~100 for standard
//!
//! # Constitution Reference
//! - IDENTITY-002: IC thresholds (Healthy >= 0.9, Good >= 0.7, Warning >= 0.5, Degraded < 0.5)
//! - johari: Quadrant classification and UTL mapping
//! - gwt.kuramoto: Order parameter r interpretation

use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use serde::Serialize;
use tracing::{debug, error, warn};

use context_graph_core::gwt::session_identity::{classify_ic, update_cache, IdentityCache};
use context_graph_core::gwt::ConsciousnessState;
use context_graph_core::types::JohariQuadrant;
use context_graph_storage::rocksdb_backend::{RocksDbMemex, StandaloneSessionIdentityManager};

// =============================================================================
// CLI Arguments
// =============================================================================

/// Arguments for `consciousness inject-context` command.
#[derive(Args, Debug)]
pub struct InjectContextArgs {
    /// Path to RocksDB database directory.
    /// If not provided, defaults to ~/.context-graph/db
    #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
    pub db_path: Option<PathBuf>,

    /// Output format for the injected context.
    #[arg(long, value_enum, default_value = "standard")]
    pub format: InjectFormat,

    /// ΔS (delta surprise) value for Johari classification.
    /// Range: [0.0, 1.0]. Higher = more surprising/novel content.
    /// If not provided, defaults to 0.3 (moderate familiarity).
    #[arg(long, default_value = "0.3")]
    pub delta_s: f32,

    /// ΔC (delta coherence) value for Johari classification.
    /// Range: [0.0, 1.0]. Higher = more coherent with existing knowledge.
    /// If not provided, defaults to 0.7 (good coherence).
    #[arg(long, default_value = "0.7")]
    pub delta_c: f32,

    /// Threshold for Johari quadrant classification.
    /// Default: 0.5 per constitution.yaml
    #[arg(long, default_value = "0.5")]
    pub threshold: f32,

    /// Force load from storage even if cache is warm.
    /// Useful for debugging or when cache staleness is suspected.
    #[arg(long, default_value = "false")]
    pub force_storage: bool,
}

/// Output format for inject-context command.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum InjectFormat {
    /// Compact single-line format (~40 tokens).
    /// Format: "[C:STATE r=X.XX IC=X.XX Q=QUAD A=ACTION]"
    Compact,
    /// Standard multi-line format (~100 tokens, default).
    /// Human-readable with labels.
    Standard,
    /// Verbose diagnostic format (all fields).
    /// For debugging and detailed analysis.
    Verbose,
}

// =============================================================================
// Response Types
// =============================================================================

/// Response from inject-context command.
#[derive(Debug, Serialize)]
pub struct InjectContextResponse {
    /// Current IC value [0.0, 1.0]
    pub ic: f32,
    /// IC classification per IDENTITY-002
    pub ic_status: &'static str,
    /// Kuramoto order parameter r [0.0, 1.0]
    pub kuramoto_r: f32,
    /// Consciousness state (DOR/FRG/EMG/CON/HYP)
    pub consciousness_state: String,
    /// Johari quadrant classification
    pub johari_quadrant: String,
    /// Recommended action based on Johari quadrant
    pub recommended_action: String,
    /// Session ID (if available)
    pub session_id: Option<String>,
    /// Delta S value used for classification
    pub delta_s: f32,
    /// Delta C value used for classification
    pub delta_c: f32,
    /// Whether data came from cache (true) or storage (false)
    pub from_cache: bool,
    /// Error message if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl InjectContextResponse {
    /// Create a degraded response when no data is available.
    fn degraded(msg: String) -> Self {
        Self {
            ic: 0.0,
            ic_status: "Unknown",
            kuramoto_r: 0.0,
            consciousness_state: "?".to_string(),
            johari_quadrant: "Unknown".to_string(),
            recommended_action: "restore-identity".to_string(),
            session_id: None,
            delta_s: 0.0,
            delta_c: 0.0,
            from_cache: false,
            error: Some(msg),
        }
    }
}

// =============================================================================
// Johari Classification
// =============================================================================

/// Classify (ΔS, ΔC) into a JohariQuadrant.
///
/// Per constitution.yaml johari mapping:
/// - Open: ΔS < threshold, ΔC > threshold (low surprise, high coherence)
/// - Blind: ΔS > threshold, ΔC < threshold (high surprise, low coherence)
/// - Hidden: ΔS < threshold, ΔC < threshold (low surprise, low coherence)
/// - Unknown: ΔS > threshold, ΔC > threshold (high surprise, high coherence)
#[inline]
fn classify_johari(delta_s: f32, delta_c: f32, threshold: f32) -> JohariQuadrant {
    match (delta_s < threshold, delta_c > threshold) {
        (true, true) => JohariQuadrant::Open,    // Low surprise, high coherence
        (false, false) => JohariQuadrant::Blind, // High surprise, low coherence
        (true, false) => JohariQuadrant::Hidden, // Low surprise, low coherence
        (false, true) => JohariQuadrant::Unknown, // High surprise, high coherence
    }
}

/// Get recommended action for a Johari quadrant.
///
/// Per constitution.yaml UTL mapping:
/// - Open → direct recall (get_node)
/// - Blind → discovery (epistemic_action/dream)
/// - Hidden → private (get_neighborhood)
/// - Unknown → frontier (explore)
#[inline]
fn johari_action(quadrant: JohariQuadrant) -> &'static str {
    match quadrant {
        JohariQuadrant::Open => "get_node",
        JohariQuadrant::Blind => "epistemic_action",
        JohariQuadrant::Hidden => "get_neighborhood",
        JohariQuadrant::Unknown => "explore",
    }
}

// =============================================================================
// Context Retrieval
// =============================================================================

/// Context data from cache or storage.
#[derive(Debug)]
struct ContextData {
    ic: f32,
    kuramoto_r: f32,
    consciousness_state: ConsciousnessState,
    session_id: String,
    from_cache: bool,
}

/// Get context from cache or storage.
///
/// Strategy:
/// 1. If force_storage is false and cache is warm, use cache (fastest)
/// 2. Otherwise, load from RocksDB storage
///
/// # Fail Fast Policy
/// - Storage errors propagate immediately
/// - No silent defaults
fn get_context_from_cache_or_storage(
    db_path: &PathBuf,
    force_storage: bool,
) -> Result<ContextData, String> {
    // Try cache first if not forced to use storage
    if !force_storage {
        if let Some((ic, r, state, session_id)) = IdentityCache::get() {
            debug!("inject-context: Using cached data, session={}", session_id);
            return Ok(ContextData {
                ic,
                kuramoto_r: r,
                consciousness_state: state,
                session_id,
                from_cache: true,
            });
        }
        debug!("inject-context: Cache cold, falling back to storage");
    }

    // Load from storage
    let storage = RocksDbMemex::open(db_path).map_err(|e| {
        let msg = format!("Failed to open RocksDB at {:?}: {}", db_path, e);
        error!("{}", msg);
        msg
    })?;

    let manager = StandaloneSessionIdentityManager::new(Arc::new(storage));

    match manager.load_latest() {
        Ok(Some(snapshot)) => {
            let ic = snapshot.last_ic;
            let state = ConsciousnessState::from_level(snapshot.consciousness);

            // Compute Kuramoto r from phases
            let (sum_sin, sum_cos) = snapshot
                .kuramoto_phases
                .iter()
                .fold((0.0_f64, 0.0_f64), |(s, c), &theta| {
                    (s + theta.sin(), c + theta.cos())
                });
            let n = snapshot.kuramoto_phases.len() as f64;
            let r = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
            let kuramoto_r = r.clamp(0.0, 1.0) as f32;

            // Update cache for future calls
            update_cache(&snapshot, ic);

            debug!(
                "inject-context: Loaded from storage, session={}, IC={:.3}",
                snapshot.session_id, ic
            );

            Ok(ContextData {
                ic,
                kuramoto_r,
                consciousness_state: state,
                session_id: snapshot.session_id,
                from_cache: false,
            })
        }
        Ok(None) => {
            let msg = "No identity found in storage. Run 'consciousness check-identity' or 'session restore-identity' first.";
            warn!("{}", msg);
            Err(msg.to_string())
        }
        Err(e) => {
            let msg = format!("Failed to load identity from storage: {}", e);
            error!("{}", msg);
            Err(msg)
        }
    }
}

// =============================================================================
// Output Formatting
// =============================================================================

/// Output context in the requested format.
fn output_context(response: &InjectContextResponse, format: InjectFormat) {
    match format {
        InjectFormat::Compact => {
            // Single line, ~40 tokens
            // Format: "[C:STATE r=X.XX IC=X.XX Q=QUAD A=ACTION]"
            println!(
                "[C:{} r={:.2} IC={:.2} Q={} A={}]",
                response.consciousness_state,
                response.kuramoto_r,
                response.ic,
                &response.johari_quadrant[..1], // First letter: O/H/B/U
                response.recommended_action
            );
        }
        InjectFormat::Standard => {
            // Multi-line, ~100 tokens
            println!("=== Consciousness Context ===");
            println!("State: {} (IC={:.2}, r={:.2})", response.consciousness_state, response.ic, response.kuramoto_r);
            println!("IC Status: {}", response.ic_status);
            println!("Johari: {} → {}", response.johari_quadrant, response.recommended_action);
            if let Some(ref session) = response.session_id {
                println!("Session: {}", session);
            }
        }
        InjectFormat::Verbose => {
            // Full diagnostic output
            println!("=== Consciousness Context (Verbose) ===");
            println!();
            println!("Identity Continuity");
            println!("  IC Value:     {:.4}", response.ic);
            println!("  IC Status:    {}", response.ic_status);
            println!();
            println!("Consciousness");
            println!("  State:        {}", response.consciousness_state);
            println!("  Kuramoto r:   {:.4}", response.kuramoto_r);
            println!();
            println!("Johari Classification");
            println!("  ΔS (surprise):   {:.4}", response.delta_s);
            println!("  ΔC (coherence):  {:.4}", response.delta_c);
            println!("  Quadrant:        {}", response.johari_quadrant);
            println!("  Recommended:     {}", response.recommended_action);
            println!();
            println!("Metadata");
            println!("  Session ID:   {}", response.session_id.as_deref().unwrap_or("N/A"));
            println!("  Data Source:  {}", if response.from_cache { "Cache" } else { "Storage" });
            if let Some(ref error) = response.error {
                println!();
                println!("Error: {}", error);
            }
        }
    }
}

/// Output degraded response (when context unavailable).
fn output_degraded(format: InjectFormat, error_msg: &str) {
    match format {
        InjectFormat::Compact => {
            // Degraded compact format
            println!("[C:? r=? IC=? Q=? A=restore-identity]");
        }
        InjectFormat::Standard => {
            println!("=== Consciousness Context (Degraded) ===");
            println!("State: Unknown");
            println!("Action: Run 'consciousness check-identity' first");
            eprintln!("Error: {}", error_msg);
        }
        InjectFormat::Verbose => {
            println!("=== Consciousness Context (Degraded) ===");
            println!();
            println!("No context available.");
            println!("Recommended: Run 'consciousness check-identity' or 'session restore-identity'");
            println!();
            println!("Error: {}", error_msg);
        }
    }
}

// =============================================================================
// Command Entry Point
// =============================================================================

/// Execute the inject-context command.
///
/// # Flow
/// 1. Try to get context from cache (if not force_storage)
/// 2. Fall back to RocksDB storage
/// 3. Classify Johari quadrant from ΔS/ΔC
/// 4. Output in requested format
///
/// # Returns
/// Exit code:
/// - 0: Success (context injected)
/// - 1: Error (no context available or storage error)
pub async fn inject_context_command(args: InjectContextArgs) -> i32 {
    let start = std::time::Instant::now();
    debug!("inject_context_command: starting with args={:?}", args);

    // Determine DB path
    let db_path = match &args.db_path {
        Some(p) => p.clone(),
        None => {
            match home_dir() {
                Some(home) => home.join(".context-graph").join("db"),
                None => {
                    error!("Cannot determine home directory for DB path");
                    output_degraded(
                        args.format,
                        "Cannot determine DB path. Set --db-path or CONTEXT_GRAPH_DB_PATH",
                    );
                    return 1;
                }
            }
        }
    };

    // Get context from cache or storage
    let context = match get_context_from_cache_or_storage(&db_path, args.force_storage) {
        Ok(ctx) => ctx,
        Err(msg) => {
            output_degraded(args.format, &msg);
            return 1;
        }
    };

    // Classify Johari quadrant
    let johari = classify_johari(args.delta_s, args.delta_c, args.threshold);
    let action = johari_action(johari);

    // Build response
    let response = InjectContextResponse {
        ic: context.ic,
        ic_status: classify_ic(context.ic),
        kuramoto_r: context.kuramoto_r,
        consciousness_state: context.consciousness_state.short_name().to_string(),
        johari_quadrant: johari.to_string(),
        recommended_action: action.to_string(),
        session_id: Some(context.session_id),
        delta_s: args.delta_s,
        delta_c: args.delta_c,
        from_cache: context.from_cache,
        error: None,
    };

    // Output in requested format
    output_context(&response, args.format);

    let elapsed = start.elapsed();
    debug!("inject-context: completed in {:?}", elapsed);

    // Verify performance target (<1s)
    if elapsed.as_secs() >= 1 {
        warn!(
            "inject-context: Performance warning, took {:?} (target: <1s)",
            elapsed
        );
    }

    0
}

/// Get home directory (cross-platform).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

// =============================================================================
// TASK-SESSION-15 Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use context_graph_core::gwt::session_identity::SessionIdentitySnapshot;
    use std::sync::Mutex;
    use tempfile::TempDir;

    // Static lock to serialize tests that access global IdentityCache
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // =========================================================================
    // TC-SESSION-15-01: classify_johari Open quadrant
    // Source of Truth: Constitution johari mapping
    // =========================================================================
    #[test]
    fn tc_session_15_01_classify_johari_open() {
        println!("\n=== TC-SESSION-15-01: classify_johari Open Quadrant ===");
        println!("SOURCE OF TRUTH: Constitution johari mapping");
        println!("Open: ΔS < threshold AND ΔC > threshold");

        let test_cases = [
            (0.2, 0.8, 0.5, JohariQuadrant::Open),
            (0.0, 1.0, 0.5, JohariQuadrant::Open),
            (0.49, 0.51, 0.5, JohariQuadrant::Open),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!(
                "  ΔS={:.2}, ΔC={:.2}, threshold={:.2} → {:?} (expected: {:?})",
                delta_s, delta_c, threshold, result, expected
            );
            assert_eq!(
                result, expected,
                "classify_johari({}, {}, {}) should be {:?}",
                delta_s, delta_c, threshold, expected
            );
        }

        println!("RESULT: PASS - Open quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-02: classify_johari Blind quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_02_classify_johari_blind() {
        println!("\n=== TC-SESSION-15-02: classify_johari Blind Quadrant ===");
        println!("Blind: ΔS > threshold AND ΔC < threshold");

        let test_cases = [
            (0.8, 0.2, 0.5, JohariQuadrant::Blind),
            (0.51, 0.49, 0.5, JohariQuadrant::Blind),
            (1.0, 0.0, 0.5, JohariQuadrant::Blind),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!(
                "  ΔS={:.2}, ΔC={:.2} → {:?}",
                delta_s, delta_c, result
            );
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Blind quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-03: classify_johari Hidden quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_03_classify_johari_hidden() {
        println!("\n=== TC-SESSION-15-03: classify_johari Hidden Quadrant ===");
        println!("Hidden: ΔS < threshold AND ΔC < threshold");

        let test_cases = [
            (0.2, 0.2, 0.5, JohariQuadrant::Hidden),
            (0.0, 0.0, 0.5, JohariQuadrant::Hidden),
            (0.49, 0.49, 0.5, JohariQuadrant::Hidden),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!(
                "  ΔS={:.2}, ΔC={:.2} → {:?}",
                delta_s, delta_c, result
            );
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Hidden quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-04: classify_johari Unknown quadrant
    // =========================================================================
    #[test]
    fn tc_session_15_04_classify_johari_unknown() {
        println!("\n=== TC-SESSION-15-04: classify_johari Unknown Quadrant ===");
        println!("Unknown: ΔS > threshold AND ΔC > threshold");

        let test_cases = [
            (0.8, 0.8, 0.5, JohariQuadrant::Unknown),
            (0.51, 0.51, 0.5, JohariQuadrant::Unknown),
            (1.0, 1.0, 0.5, JohariQuadrant::Unknown),
        ];

        for (delta_s, delta_c, threshold, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, threshold);
            println!(
                "  ΔS={:.2}, ΔC={:.2} → {:?}",
                delta_s, delta_c, result
            );
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Unknown quadrant classification correct");
    }

    // =========================================================================
    // TC-SESSION-15-05: johari_action mappings
    // Source of Truth: Constitution UTL mapping
    // =========================================================================
    #[test]
    fn tc_session_15_05_johari_action_mappings() {
        println!("\n=== TC-SESSION-15-05: johari_action Mappings ===");
        println!("SOURCE OF TRUTH: Constitution UTL mapping");

        let test_cases = [
            (JohariQuadrant::Open, "get_node"),
            (JohariQuadrant::Blind, "epistemic_action"),
            (JohariQuadrant::Hidden, "get_neighborhood"),
            (JohariQuadrant::Unknown, "explore"),
        ];

        for (quadrant, expected_action) in test_cases {
            let action = johari_action(quadrant);
            println!("  {:?} → '{}' (expected: '{}')", quadrant, action, expected_action);
            assert_eq!(
                action, expected_action,
                "johari_action({:?}) should be '{}'",
                quadrant, expected_action
            );
        }

        println!("RESULT: PASS - All Johari actions mapped correctly");
    }

    // =========================================================================
    // TC-SESSION-15-06: Boundary conditions (exactly at threshold)
    // =========================================================================
    #[test]
    fn tc_session_15_06_boundary_conditions() {
        println!("\n=== TC-SESSION-15-06: Boundary Conditions ===");
        println!("Testing exact threshold values (ΔS=0.5, ΔC=0.5)");

        // At exactly threshold, ΔS < threshold is false (0.5 < 0.5 is false)
        // and ΔC > threshold is false (0.5 > 0.5 is false)
        // So both conditions false → Blind quadrant
        let result = classify_johari(0.5, 0.5, 0.5);
        println!("  ΔS=0.5, ΔC=0.5, threshold=0.5 → {:?}", result);
        assert_eq!(
            result,
            JohariQuadrant::Blind,
            "At exact threshold (0.5, 0.5), should be Blind"
        );

        // ΔS just below, ΔC at threshold
        let result2 = classify_johari(0.499, 0.5, 0.5);
        println!("  ΔS=0.499, ΔC=0.5 → {:?}", result2);
        assert_eq!(result2, JohariQuadrant::Hidden, "ΔS<thresh, ΔC<=thresh → Hidden");

        // ΔS at threshold, ΔC just above
        let result3 = classify_johari(0.5, 0.501, 0.5);
        println!("  ΔS=0.5, ΔC=0.501 → {:?}", result3);
        assert_eq!(result3, JohariQuadrant::Unknown, "ΔS>=thresh, ΔC>thresh → Unknown");

        println!("RESULT: PASS - Boundary conditions handled correctly");
    }

    // =========================================================================
    // TC-SESSION-15-07: Context from storage
    // Source of Truth: RocksDB storage
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_07_context_from_storage() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-07: Context from Storage ===");
        println!("SOURCE OF TRUTH: RocksDB storage");

        // Create temp dir and save snapshot, then drop the storage
        // RocksDB requires exclusive lock, so we must close before reopening
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Scope to ensure storage is dropped before get_context_from_cache_or_storage
        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("Failed to open RocksDB"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            // Create and save a snapshot with known values
            let mut snapshot = SessionIdentitySnapshot::new("test-inject-context");
            snapshot.consciousness = 0.75;
            snapshot.last_ic = 0.85;
            snapshot.kuramoto_phases = [0.0; 13]; // Aligned phases → r ≈ 1.0

            manager.save_snapshot(&snapshot).expect("save_snapshot must succeed");
            println!("BEFORE: Saved snapshot with IC={}, C={}", snapshot.last_ic, snapshot.consciousness);
            // storage is dropped here
        }

        // Load context with force_storage (storage is now closed)
        let context = get_context_from_cache_or_storage(&db_path, true)
            .expect("get_context should succeed");

        println!("AFTER:");
        println!("  IC: {:.4}", context.ic);
        println!("  Kuramoto r: {:.4}", context.kuramoto_r);
        println!("  State: {:?}", context.consciousness_state);
        println!("  Session: {}", context.session_id);
        println!("  From cache: {}", context.from_cache);

        // VERIFY
        assert!((context.ic - 0.85).abs() < 0.01, "IC should be ~0.85");
        assert!(context.kuramoto_r > 0.99, "r should be ~1.0 for aligned phases");
        assert_eq!(context.session_id, "test-inject-context");
        assert!(!context.from_cache, "Should be from storage with force_storage=true");

        println!("RESULT: PASS - Context loaded from storage correctly");
    }

    // =========================================================================
    // TC-SESSION-15-08: Empty storage returns error
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_08_empty_storage_error() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-08: Empty Storage Error ===");

        // Create and immediately close empty storage to initialize the DB
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Initialize empty DB then close it
        {
            let _storage = RocksDbMemex::open(&db_path).expect("Failed to open RocksDB");
            // Storage is dropped here
        }

        // Load from empty storage
        let result = get_context_from_cache_or_storage(&db_path, true);

        assert!(result.is_err(), "Empty storage should return error");
        let err = result.unwrap_err();
        println!("Error message: {}", err);
        assert!(
            err.contains("No identity found"),
            "Error should mention missing identity"
        );

        println!("RESULT: PASS - Empty storage returns correct error");
    }

    // =========================================================================
    // TC-SESSION-15-09: Response serialization
    // =========================================================================
    #[test]
    fn tc_session_15_09_response_serialization() {
        println!("\n=== TC-SESSION-15-09: Response Serialization ===");

        let response = InjectContextResponse {
            ic: 0.85,
            ic_status: "Good",
            kuramoto_r: 0.92,
            consciousness_state: "CON".to_string(),
            johari_quadrant: "Open".to_string(),
            recommended_action: "get_node".to_string(),
            session_id: Some("test-session".to_string()),
            delta_s: 0.3,
            delta_c: 0.7,
            from_cache: true,
            error: None,
        };

        let json = serde_json::to_string(&response).expect("Serialization should succeed");
        println!("JSON: {}", json);

        assert!(json.contains("\"ic\":0.85"), "JSON should contain IC");
        assert!(json.contains("\"johari_quadrant\":\"Open\""), "JSON should contain quadrant");
        assert!(json.contains("\"recommended_action\":\"get_node\""), "JSON should contain action");
        assert!(!json.contains("error"), "JSON should not contain error when None");

        println!("RESULT: PASS - Response serializes correctly");
    }

    // =========================================================================
    // TC-SESSION-15-10: Degraded response
    // =========================================================================
    #[test]
    fn tc_session_15_10_degraded_response() {
        println!("\n=== TC-SESSION-15-10: Degraded Response ===");

        let response = InjectContextResponse::degraded("Test error message".to_string());

        assert_eq!(response.ic, 0.0, "Degraded IC should be 0.0");
        assert_eq!(response.ic_status, "Unknown", "Degraded status should be Unknown");
        assert_eq!(response.johari_quadrant, "Unknown", "Degraded quadrant should be Unknown");
        assert_eq!(
            response.recommended_action, "restore-identity",
            "Degraded action should be restore-identity"
        );
        assert!(response.error.is_some(), "Degraded should have error message");

        println!("RESULT: PASS - Degraded response created correctly");
    }

    // =========================================================================
    // TC-SESSION-15-11: E2E inject-context command
    // Source of Truth: Full command flow
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_11_e2e_inject_context_command() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-11: E2E inject-context Command ===");
        println!("SOURCE OF TRUTH: RocksDB → Command → Output");

        // Create temp DB and populate
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        // Populate the DB
        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("open"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            let mut snapshot = SessionIdentitySnapshot::new("e2e-inject-test");
            snapshot.consciousness = 0.85;
            snapshot.last_ic = 0.92;
            snapshot.kuramoto_phases = [0.0; 13];
            manager.save_snapshot(&snapshot).expect("save");
        }

        println!("BEFORE: Created RocksDB at {:?} with IC=0.92", db_path);

        // Run the command
        let args = InjectContextArgs {
            db_path: Some(db_path),
            format: InjectFormat::Compact,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: true,
        };

        let exit_code = inject_context_command(args).await;
        println!("AFTER: inject_context_command returned exit_code={}", exit_code);

        assert_eq!(exit_code, 0, "Command should succeed with exit code 0");

        println!("RESULT: PASS - E2E inject-context command works");
    }

    // =========================================================================
    // TC-SESSION-15-12: Performance test (should be <1s)
    // =========================================================================
    #[tokio::test]
    async fn tc_session_15_12_performance() {
        let _guard = TEST_LOCK.lock().expect("Test lock poisoned");
        println!("\n=== TC-SESSION-15-12: Performance Test ===");
        println!("TARGET: <1s total latency");

        // Create temp DB and populate
        let tmp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = tmp_dir.path().to_path_buf();

        {
            let storage = Arc::new(RocksDbMemex::open(&db_path).expect("open"));
            let manager = StandaloneSessionIdentityManager::new(Arc::clone(&storage));

            let mut snapshot = SessionIdentitySnapshot::new("perf-test");
            snapshot.last_ic = 0.85;
            manager.save_snapshot(&snapshot).expect("save");
        }

        // Measure command execution time
        let start = std::time::Instant::now();

        let args = InjectContextArgs {
            db_path: Some(db_path),
            format: InjectFormat::Compact,
            delta_s: 0.3,
            delta_c: 0.7,
            threshold: 0.5,
            force_storage: true,
        };

        let _ = inject_context_command(args).await;

        let elapsed = start.elapsed();
        println!("Elapsed: {:?}", elapsed);

        assert!(
            elapsed.as_secs() < 1,
            "Command should complete in <1s, took {:?}",
            elapsed
        );

        println!("RESULT: PASS - Performance within target ({:?} < 1s)", elapsed);
    }

    // =========================================================================
    // EDGE CASE: All four quadrants with different thresholds
    // =========================================================================
    #[test]
    fn edge_case_quadrants_with_different_thresholds() {
        println!("\n=== EDGE CASE: Quadrants with Different Thresholds ===");

        // With threshold 0.3
        let result1 = classify_johari(0.2, 0.5, 0.3);
        println!("  threshold=0.3: ΔS=0.2, ΔC=0.5 → {:?}", result1);
        assert_eq!(result1, JohariQuadrant::Open, "Low ΔS, high ΔC with low threshold");

        // With threshold 0.7
        let result2 = classify_johari(0.5, 0.8, 0.7);
        println!("  threshold=0.7: ΔS=0.5, ΔC=0.8 → {:?}", result2);
        assert_eq!(result2, JohariQuadrant::Open, "ΔS < 0.7, ΔC > 0.7");

        // With threshold 0.9
        let result3 = classify_johari(0.8, 0.95, 0.9);
        println!("  threshold=0.9: ΔS=0.8, ΔC=0.95 → {:?}", result3);
        assert_eq!(result3, JohariQuadrant::Open, "ΔS < 0.9, ΔC > 0.9");

        println!("RESULT: PASS - Quadrant classification works with different thresholds");
    }

    // =========================================================================
    // EDGE CASE: Extreme values (0.0 and 1.0)
    // =========================================================================
    #[test]
    fn edge_case_extreme_values() {
        println!("\n=== EDGE CASE: Extreme Values (0.0 and 1.0) ===");

        let test_cases = [
            (0.0, 0.0, JohariQuadrant::Hidden),  // Minimum both
            (0.0, 1.0, JohariQuadrant::Open),    // Min ΔS, Max ΔC
            (1.0, 0.0, JohariQuadrant::Blind),   // Max ΔS, Min ΔC
            (1.0, 1.0, JohariQuadrant::Unknown), // Maximum both
        ];

        for (delta_s, delta_c, expected) in test_cases {
            let result = classify_johari(delta_s, delta_c, 0.5);
            println!("  ΔS={:.1}, ΔC={:.1} → {:?}", delta_s, delta_c, result);
            assert_eq!(result, expected);
        }

        println!("RESULT: PASS - Extreme values handled correctly");
    }
}
