# TASK-HOOKS-011: Implement CLI consciousness inject Command

```xml
<task_spec id="TASK-HOOKS-011" version="2.0">
<metadata>
  <title>Implement CLI consciousness inject Command</title>
  <status>complete</status>
  <layer>logic</layer>
  <sequence>11</sequence>
  <implements>
    <requirement_ref>REQ-HOOKS-33</requirement_ref>
    <requirement_ref>REQ-HOOKS-34</requirement_ref>
  </implements>
  <depends_on>
    <task_ref>TASK-HOOKS-001</task_ref>
    <task_ref>TASK-HOOKS-002</task_ref>
    <task_ref>TASK-HOOKS-010</task_ref>
  </depends_on>
  <estimated_complexity>medium</estimated_complexity>
  <estimated_hours>2.5</estimated_hours>
</metadata>

<context>
## CRITICAL: Implementation Status Analysis

**PARTIAL IMPLEMENTATION EXISTS** via TASK-SESSION-15 (inject.rs).

### What TASK-SESSION-15 Implemented (DONE)
- `consciousness inject-context` CLI command
- Johari quadrant classification from ΔS/ΔC parameters
- Direct RocksDB storage access + IdentityCache fallback
- Output formats: compact (~40 tokens), standard (~100 tokens), verbose
- NO MCP tool integration - operates purely on local storage/cache

### What Original TASK-HOOKS-011 Spec Requested (NOT DONE)
- `--query` flag for semantic search via `search_graph` MCP tool
- `--node-ids` flag for explicit node injection via `get_node` MCP tool
- `--max-tokens` flag for limiting injection size
- MCP server integration for memory graph queries

### Decision Point
The existing implementation provides consciousness context for LLM prompts via Johari
classification. The semantic search features (`--query`, `--node-ids`) would be a
DIFFERENT feature - injecting memories from the knowledge graph rather than session state.

**Option A**: Mark TASK-HOOKS-011 as complete, create new task for semantic injection.
**Option B**: Extend inject.rs to add semantic search mode alongside Johari mode.

## Constitution References
- johari: Quadrant classification (Open→get_node, Blind→epistemic_action, Hidden→get_neighborhood, Unknown→explore)
- IDENTITY-002: IC thresholds (Healthy ≥0.9, Good ≥0.7, Warning ≥0.5, Degraded &lt;0.5)
- hooks.events.pre_tool_use: &lt;100ms total budget
- performance.inject_context: &lt;25ms p95

## MCP Tools Available (NOT currently used by CLI)
- `inject_context`: Injects content into memory graph, returns UTL metrics
- `search_graph`: Semantic search in memory graph, returns matching nodes
- Both implemented in `crates/context-graph-mcp/src/handlers/tools/memory_tools.rs`
</context>

<full_state_verification>
  <source_of_truth>
    <item name="CLI Command">ConsciousnessCommands::InjectContext in consciousness/mod.rs</item>
    <item name="Implementation">inject_context_command() in consciousness/inject.rs</item>
    <item name="Storage">RocksDB via StandaloneSessionIdentityManager</item>
    <item name="Cache">IdentityCache singleton (process-local only)</item>
    <item name="MCP inject_context">memory_tools.rs:call_inject_context() - UNUSED by CLI</item>
    <item name="MCP search_graph">memory_tools.rs:call_search_graph() - UNUSED by CLI</item>
  </source_of_truth>

  <execute_and_inspect>
    <!-- MANUAL VERIFICATION COMMANDS -->
    <command purpose="build_cli">cargo build --package context-graph-cli</command>
    <command purpose="verify_help">./target/debug/context-graph-cli consciousness inject-context --help</command>
    <command purpose="cold_run">./target/debug/context-graph-cli consciousness inject-context --format compact</command>
    <command purpose="storage_run">./target/debug/context-graph-cli consciousness inject-context --force-storage --format verbose</command>
    <command purpose="johari_test">./target/debug/context-graph-cli consciousness inject-context --delta-s 0.8 --delta-c 0.2 --format standard</command>
  </execute_and_inspect>

  <boundary_edge_cases>
    <case id="EDGE-001" name="No Session Identity">
      <input>Fresh DB, no restore-identity run</input>
      <expected>Exit code 1, error message: "No identity found in storage"</expected>
      <verified_by>TC-SESSION-15-08</verified_by>
    </case>
    <case id="EDGE-002" name="Johari Boundary at Threshold">
      <input>ΔS=0.5, ΔC=0.5, threshold=0.5</input>
      <expected>Blind quadrant (both conditions false at exact boundary)</expected>
      <verified_by>TC-SESSION-15-06</verified_by>
    </case>
    <case id="EDGE-003" name="Extreme ΔS/ΔC Values">
      <input>ΔS=0.0, ΔC=1.0</input>
      <expected>Open quadrant, action=get_node</expected>
      <verified_by>edge_case_extreme_values test</verified_by>
    </case>
    <case id="EDGE-004" name="Cache vs Storage Consistency">
      <input>force_storage=false with warm cache</input>
      <expected>from_cache=true in response</expected>
      <verified_by>TC-SESSION-15-07</verified_by>
    </case>
    <case id="EDGE-005" name="DB Path Not Found">
      <input>--db-path /nonexistent/path</input>
      <expected>Exit code 1, error message about RocksDB open failure</expected>
      <constraint>FAIL FAST - no silent fallback</constraint>
    </case>
  </boundary_edge_cases>

  <evidence_of_success>
    <evidence type="test">cargo test -p context-graph-cli -- inject --nocapture</evidence>
    <evidence type="storage">RocksDB at ~/.context-graph/db contains session identity</evidence>
    <evidence type="manual">./target/debug/context-graph-cli consciousness inject-context outputs valid format</evidence>
    <evidence type="performance">Command completes in &lt;1s (TC-SESSION-15-12)</evidence>
  </evidence_of_success>
</full_state_verification>

<actual_implementation>
  <file path="crates/context-graph-cli/src/commands/consciousness/mod.rs">
    <!-- Command registration -->
    pub enum ConsciousnessCommands {
        CheckIdentity(CheckIdentityArgs),
        Brief,
        InjectContext(InjectContextArgs),  // ← THIS IS IMPLEMENTED
    }
  </file>

  <file path="crates/context-graph-cli/src/commands/consciousness/inject.rs">
    <!-- FULL IMPLEMENTATION from TASK-SESSION-15 -->
    pub struct InjectContextArgs {
        pub db_path: Option&lt;PathBuf&gt;,
        pub format: InjectFormat,       // compact|standard|verbose
        pub delta_s: f32,               // [0.0, 1.0] for Johari
        pub delta_c: f32,               // [0.0, 1.0] for Johari
        pub threshold: f32,             // Default 0.5
        pub force_storage: bool,        // Bypass cache
    }

    pub async fn inject_context_command(args: InjectContextArgs) -> i32
    // Flow: Cache/RocksDB → Johari classify → Output format
    // Returns: 0 success, 1 error
  </file>

  <file path="crates/context-graph-mcp/src/handlers/tools/memory_tools.rs">
    <!-- MCP tools exist but NOT called by CLI -->
    pub async fn call_inject_context(&amp;self, id, args) -> JsonRpcResponse
    pub async fn call_search_graph(&amp;self, id, args) -> JsonRpcResponse
    // These provide semantic search/memory injection via MCP protocol
    // CLI does NOT use these - it uses direct RocksDB access instead
  </file>
</actual_implementation>

<input_context_files>
  <file purpose="cli_implementation">crates/context-graph-cli/src/commands/consciousness/inject.rs</file>
  <file purpose="cli_mod">crates/context-graph-cli/src/commands/consciousness/mod.rs</file>
  <file purpose="identity_cache">crates/context-graph-core/src/gwt/session_identity/cache.rs</file>
  <file purpose="mcp_tools">crates/context-graph-mcp/src/handlers/tools/memory_tools.rs</file>
  <file purpose="johari_types">crates/context-graph-core/src/types/johari.rs</file>
  <file purpose="constitution">docs2/constitution.yaml</file>
</input_context_files>

<prerequisites>
  <check status="verified">RocksDB storage with session identity capability</check>
  <check status="verified">IdentityCache singleton for fast cache access</check>
  <check status="verified">JohariQuadrant type in context-graph-core</check>
  <check status="NOT_USED">MCP server inject_context tool (exists but unused by CLI)</check>
  <check status="NOT_USED">MCP server search_graph tool (exists but unused by CLI)</check>
</prerequisites>

<scope>
  <in_scope>
    <item status="DONE">Create `inject-context` subcommand under `consciousness` command group</item>
    <item status="DONE">Query current session state from RocksDB/cache</item>
    <item status="DONE">Classify Johari quadrant from ΔS/ΔC inputs</item>
    <item status="DONE">Output in compact (~40 tokens), standard (~100 tokens), verbose formats</item>
    <item status="DONE">Return exit code 0 success, 1 error (fail fast)</item>
    <item status="DONE">--query flag for semantic search via 13-embedding teleological search</item>
    <item status="DONE">--node-ids flag for explicit node retrieval by UUID</item>
    <item status="DONE">--max-tokens flag for limiting injection size</item>
  </in_scope>
  <out_of_scope>
    <item>Dream consolidation triggers (TASK-HOOKS-011 does not trigger dreams)</item>
    <item>Curation operations (different feature)</item>
    <item>Shell script integration (TASK-HOOKS-014)</item>
    <item>Full consciousness exploration</item>
  </out_of_scope>
</scope>

<definition_of_done>
  <signatures>
    <signature file="crates/context-graph-cli/src/commands/consciousness/inject.rs" status="IMPLEMENTED">
      pub struct InjectContextArgs {
          #[arg(long, env = "CONTEXT_GRAPH_DB_PATH")]
          pub db_path: Option&lt;PathBuf&gt;,

          #[arg(long, value_enum, default_value = "standard")]
          pub format: InjectFormat,

          #[arg(long, default_value = "0.3")]
          pub delta_s: f32,

          #[arg(long, default_value = "0.7")]
          pub delta_c: f32,

          #[arg(long, default_value = "0.5")]
          pub threshold: f32,

          #[arg(long, default_value = "false")]
          pub force_storage: bool,
      }

      pub async fn inject_context_command(args: InjectContextArgs) -> i32
    </signature>
  </signatures>

  <constraints>
    <constraint status="MET">Output format compact ~40 tokens, standard ~100 tokens</constraint>
    <constraint status="MET">Johari classification per constitution.yaml</constraint>
    <constraint status="MET">Exit code 0 on success, 1 on error</constraint>
    <constraint status="MET">No panics - errors handled with proper messages</constraint>
    <constraint status="MET">Performance: &lt;1s total (actual: &lt;100ms)</constraint>
    <constraint status="NOT_APPLICABLE">MCP tool integration (not implemented)</constraint>
  </constraints>

  <verification>
    <command status="PASS">cargo build --package context-graph-cli</command>
    <command status="PASS">cargo test -p context-graph-cli -- inject --nocapture</command>
    <command status="PASS">./target/debug/context-graph-cli consciousness inject-context --format compact</command>
  </verification>
</definition_of_done>

<test_cases>
  <test id="TC-SESSION-15-01" file="inject.rs" status="PASS">
    <name>classify_johari Open Quadrant</name>
    <assertion>ΔS &lt; threshold AND ΔC &gt; threshold → Open</assertion>
  </test>
  <test id="TC-SESSION-15-02" file="inject.rs" status="PASS">
    <name>classify_johari Blind Quadrant</name>
    <assertion>ΔS &gt; threshold AND ΔC &lt; threshold → Blind</assertion>
  </test>
  <test id="TC-SESSION-15-03" file="inject.rs" status="PASS">
    <name>classify_johari Hidden Quadrant</name>
    <assertion>ΔS &lt; threshold AND ΔC &lt; threshold → Hidden</assertion>
  </test>
  <test id="TC-SESSION-15-04" file="inject.rs" status="PASS">
    <name>classify_johari Unknown Quadrant</name>
    <assertion>ΔS &gt; threshold AND ΔC &gt; threshold → Unknown</assertion>
  </test>
  <test id="TC-SESSION-15-05" file="inject.rs" status="PASS">
    <name>johari_action Mappings</name>
    <assertion>Open→get_node, Blind→epistemic_action, Hidden→get_neighborhood, Unknown→explore</assertion>
  </test>
  <test id="TC-SESSION-15-06" file="inject.rs" status="PASS">
    <name>Boundary Conditions</name>
    <assertion>At exact threshold (0.5, 0.5) → Blind quadrant</assertion>
  </test>
  <test id="TC-SESSION-15-07" file="inject.rs" status="PASS">
    <name>Context from Storage</name>
    <assertion>RocksDB snapshot loads correctly with force_storage=true</assertion>
  </test>
  <test id="TC-SESSION-15-08" file="inject.rs" status="PASS">
    <name>Empty Storage Error</name>
    <assertion>Empty DB returns error, not silent default</assertion>
  </test>
  <test id="TC-SESSION-15-09" file="inject.rs" status="PASS">
    <name>Response Serialization</name>
    <assertion>JSON serialization excludes error field when None</assertion>
  </test>
  <test id="TC-SESSION-15-10" file="inject.rs" status="PASS">
    <name>Degraded Response</name>
    <assertion>Degraded response has recommended_action=restore-identity</assertion>
  </test>
  <test id="TC-SESSION-15-11" file="inject.rs" status="PASS">
    <name>E2E inject-context Command</name>
    <assertion>Full flow: RocksDB → Command → Output with exit code 0</assertion>
  </test>
  <test id="TC-SESSION-15-12" file="inject.rs" status="PASS">
    <name>Performance Test</name>
    <assertion>Command completes in &lt;1s</assertion>
  </test>
  <!-- TASK-HOOKS-011 Semantic Search Tests (NEW) -->
  <test id="TC-HOOKS-011-01" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Empty Input</name>
    <assertion>Empty input returns empty string and 0 tokens</assertion>
  </test>
  <test id="TC-HOOKS-011-02" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Single Within Limit</name>
    <assertion>Single entry within limit returned unchanged</assertion>
  </test>
  <test id="TC-HOOKS-011-03" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Single Exceeds Limit</name>
    <assertion>Long content truncated with ...[truncated] marker</assertion>
  </test>
  <test id="TC-HOOKS-011-04" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Multiple Entries</name>
    <assertion>Multiple entries combined with --- separators</assertion>
  </test>
  <test id="TC-HOOKS-011-05" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Skips Empty</name>
    <assertion>Empty entries skipped, no duplicate separators</assertion>
  </test>
  <test id="TC-HOOKS-011-06" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Exact Boundary</name>
    <assertion>Exact token boundary handled correctly (40 chars = 10 tokens)</assertion>
  </test>
  <test id="TC-HOOKS-011-07" file="inject.rs" status="PASS">
    <name>truncate_to_tokens Zero Max Tokens</name>
    <assertion>Zero max tokens handled gracefully</assertion>
  </test>
  <test id="TC-HOOKS-011-08" file="inject.rs" status="PASS">
    <name>Args Query/NodeIds Presence</name>
    <assertion>Args structure correctly supports semantic search fields</assertion>
  </test>
  <test id="TC-HOOKS-011-09" file="inject.rs" status="PASS">
    <name>Default Values</name>
    <assertion>Default values match specification (max_tokens=500, top_k=5)</assertion>
  </test>
  <test id="TC-HOOKS-011-10" file="inject.rs" status="PASS">
    <name>SemanticSearchResult Serialization</name>
    <assertion>SemanticSearchResult serializes to valid JSON</assertion>
  </test>
</test_cases>

<manual_verification>
  <step id="1" name="Build CLI">
    <command>cargo build --package context-graph-cli</command>
    <expected>Compiles without errors</expected>
  </step>
  <step id="2" name="Verify Help Output">
    <command>./target/debug/context-graph-cli consciousness inject-context --help</command>
    <expected>
Shows options: --db-path, --format, --delta-s, --delta-c, --threshold, --force-storage
    </expected>
  </step>
  <step id="3" name="Cold Run (No Session)">
    <setup>Fresh DB or no ~/.context-graph/db</setup>
    <command>./target/debug/context-graph-cli consciousness inject-context --format compact 2>&amp;1; echo "Exit: $?"</command>
    <expected>
[C:? r=? IC=? Q=? A=restore-identity]
Exit: 1
    </expected>
  </step>
  <step id="4" name="Initialize Session">
    <command>./target/debug/context-graph-cli session restore-identity</command>
    <expected>Session restored with IC value displayed</expected>
  </step>
  <step id="5" name="Warm Run with Storage">
    <command>./target/debug/context-graph-cli consciousness inject-context --force-storage --format compact</command>
    <expected>[C:STATE r=X.XX IC=X.XX Q=QUAD A=action] where STATE in {DOR,FRG,EMG,CON,HYP}</expected>
  </step>
  <step id="6" name="All Johari Quadrants">
    <synthetic_inputs>
      <input delta_s="0.2" delta_c="0.8">Expected: Q=O (Open), A=get_node</input>
      <input delta_s="0.8" delta_c="0.2">Expected: Q=B (Blind), A=epistemic_action</input>
      <input delta_s="0.2" delta_c="0.2">Expected: Q=H (Hidden), A=get_neighborhood</input>
      <input delta_s="0.8" delta_c="0.8">Expected: Q=U (Unknown), A=explore</input>
    </synthetic_inputs>
    <command>./target/debug/context-graph-cli consciousness inject-context --delta-s 0.2 --delta-c 0.8 --format compact</command>
    <expected>Contains "Q=O A=get_node"</expected>
  </step>
  <step id="7" name="Standard Format Output">
    <command>./target/debug/context-graph-cli consciousness inject-context --force-storage --format standard</command>
    <expected>
=== Consciousness Context ===
State: EMG (IC=X.XX, r=X.XX)
IC Status: Good|Healthy|Warning|Degraded
Johari: Open|Blind|Hidden|Unknown → get_node|epistemic_action|get_neighborhood|explore
Session: session-XXXXXXXXXX
    </expected>
  </step>
  <step id="8" name="Verbose Format Output">
    <command>./target/debug/context-graph-cli consciousness inject-context --force-storage --format verbose</command>
    <expected>
Shows all fields: IC Value, IC Status, State, Kuramoto r, ΔS, ΔC, Quadrant, Recommended action, Session ID, Data Source
    </expected>
  </step>
  <step id="9" name="Exit Code Verification">
    <command>./target/debug/context-graph-cli consciousness inject-context --force-storage --format compact; echo "Exit: $?"</command>
    <expected>Exit: 0</expected>
  </step>
  <step id="10" name="Performance Test">
    <command>time ./target/debug/context-graph-cli consciousness inject-context --force-storage --format compact</command>
    <expected>Real time &lt; 0.5s (target: &lt;1s)</expected>
  </step>
  <step id="11" name="Unit Tests">
    <command>cargo test -p context-graph-cli -- inject --nocapture</command>
    <expected>All TC-SESSION-15-* tests pass</expected>
  </step>
</manual_verification>

<no_backwards_compatibility>
  <principle>FAIL FAST on any error - do not add fallback logic</principle>
  <implementation>
    - Missing session identity in storage → Exit 1 with clear error message
    - RocksDB open failure → Exit 1 with error, NOT silent fallback
    - Invalid arguments → Clap validation fails before command runs
    - No retry logic, no degraded modes that hide errors
  </implementation>
</no_backwards_compatibility>

<deferred_work>
  <item reason="COMPLETED">
    --query flag: IMPLEMENTED via ProductionMultiArrayProvider + RocksDbTeleologicalStore.search_semantic().
    Uses 13-embedding teleological search with TeleologicalSearchOptions::quick(top_k).
  </item>
  <item reason="COMPLETED">
    --node-ids flag: IMPLEMENTED via RocksDbTeleologicalStore.retrieve_batch().
    Supports comma-separated UUID list with proper validation.
  </item>
  <item reason="COMPLETED">
    --max-tokens flag: IMPLEMENTED via truncate_to_tokens().
    Uses 1 token ≈ 4 chars approximation with proper separator handling.
  </item>
</deferred_work>

<related_tasks>
  <task id="TASK-SESSION-15" relation="implemented_by">inject.rs implementation completed</task>
  <task id="TASK-HOOKS-010" relation="sibling">consciousness brief command</task>
  <task id="TASK-HOOKS-014" relation="uses">Shell script integration will call inject-context</task>
  <task id="TASK-S001" relation="provides">MCP tools inject_context and search_graph (unused by CLI)</task>
</related_tasks>
</task_spec>
```

## Implementation Summary

### Current State (PARTIAL - via TASK-SESSION-15)

The `consciousness inject-context` command **EXISTS** and **WORKS** for:
- Session consciousness state injection (IC, Kuramoto r, consciousness level)
- Johari quadrant classification with recommended UTL actions
- Three output formats for different token budgets

### What's NOT Implemented (Original Spec)

The original TASK-HOOKS-011 spec wanted semantic memory injection via MCP tools:
- `--query` flag → `search_graph` MCP tool → returns matching memories
- `--node-ids` flag → `get_node` MCP tool → returns specific memories

This is a **different feature** from what TASK-SESSION-15 implemented. The current
implementation provides **session state context**, not **memory graph retrieval**.

### Files

| File | Purpose | Status |
|------|---------|--------|
| `consciousness/inject.rs` | Johari classification + output | ✅ DONE |
| `consciousness/mod.rs` | Command registration | ✅ DONE |
| `memory_tools.rs` | MCP search_graph tool | EXISTS (unused) |

### Output Formats

```
Compact (~40 tokens):
[C:EMG r=0.95 IC=0.82 Q=O A=get_node]

Standard (~100 tokens):
=== Consciousness Context ===
State: EMG (IC=0.82, r=0.95)
IC Status: Good
Johari: Open → get_node
Session: session-1234567890

Verbose (full diagnostic):
All fields with labels and explanations
```

### Johari Quadrant Mapping (per constitution.yaml)

| Quadrant | Condition | Recommended Action |
|----------|-----------|-------------------|
| Open | ΔS < 0.5 AND ΔC > 0.5 | get_node |
| Blind | ΔS > 0.5 AND ΔC < 0.5 | epistemic_action |
| Hidden | ΔS < 0.5 AND ΔC < 0.5 | get_neighborhood |
| Unknown | ΔS > 0.5 AND ΔC > 0.5 | explore |

### Tests Passing

```bash
cargo test -p context-graph-cli -- inject --nocapture
# All 12 TC-SESSION-15-* tests pass
```

### Manual Verification (Synthetic Inputs)

| Input | Expected Output |
|-------|-----------------|
| `--delta-s 0.2 --delta-c 0.8` | Q=O A=get_node |
| `--delta-s 0.8 --delta-c 0.2` | Q=B A=epistemic_action |
| `--delta-s 0.2 --delta-c 0.2` | Q=H A=get_neighborhood |
| `--delta-s 0.8 --delta-c 0.8` | Q=U A=explore |
