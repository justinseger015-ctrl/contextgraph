# Claude Flow V3 - Optimal Usage Reference

## Critical Rules
- **99.9% Sequential** - Parallel only for read-only ops
- **AgentDB Required** - 88% vs 60% success rate
- **Forward-Looking Prompts** - Tell agents what next agents need
- **Never Root Files** - Use subdirectories (`/src`, `/tests`, `/docs`)
- **Batch Operations** - All related ops in ONE message
- **Memory Positional Args** - `store "key" '{"data":"val"}' --namespace "ns"` (NOT `--key`)

## Architecture
```
Claude Code = EXECUTION (Task tool, files, bash, git)
Claude Flow = COORDINATION (swarm topology, memory, neural, MCP tools)
```
MCP coordinates strategy → Task tool spawns real working agents

## Quick Start
```bash
npx claude-flow@v3alpha init                    # Initialize
npx claude-flow@v3alpha doctor --fix            # Health check
npx claude-flow@v3alpha daemon start            # Background workers
npx claude-flow@v3alpha mcp start               # MCP server
npx claude-flow@v3alpha hooks pretrain          # Bootstrap intelligence
```

## Commands

| Cmd | Key Subcommands | Purpose |
|-----|-----------------|---------|
| `init` | `wizard`, `--full`, `--minimal` | Project setup |
| `agent` | `spawn -t TYPE`, `list`, `status`, `stop`, `metrics` | Agent lifecycle |
| `swarm` | `init --v3-mode`, `start`, `status`, `scale`, `coordinate` | Multi-agent coord |
| `memory` | `store`, `retrieve`, `search -q`, `stats`, `cleanup`, `compress` | HNSW vector storage |
| `task` | `create`, `list`, `status`, `assign`, `cancel`, `retry` | Task management |
| `session` | `save`, `restore`, `list`, `export`, `import` | State persistence |
| `mcp` | `start`, `stop`, `tools`, `exec` | MCP server |
| `hooks` | `route`, `pretrain`, `pre/post-edit`, `session-*`, `worker` | Intelligence |
| `daemon` | `start`, `stop`, `status`, `trigger -w WORKER` | Background workers |
| `doctor` | `--fix`, `--install`, `-c COMPONENT` | Diagnostics |

## Agents (54+)

| Category | Agents | Use |
|----------|--------|-----|
| Core | `coder`, `reviewer`, `tester`, `planner`, `researcher` | Daily dev |
| Coord | `hierarchical-coordinator` (4-6), `mesh-coordinator` (7+), `adaptive-coordinator` | Swarm lead |
| Security | `security-architect`, `security-auditor` | Audits |
| Perf | `perf-analyzer`, `performance-engineer` | Optimization |
| GitHub | `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager` | Repo ops |
| SPARC | `sparc-coord`, `specification`, `pseudocode`, `architecture`, `refinement` | Methodology |
| Consensus | `byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `crdt-synchronizer` | Fault tolerance |

### Selection Matrix

| Task | Agent | Topology |
|------|-------|----------|
| Bug fix | `coder` + `tester` | mesh |
| Feature | `architect` + `coder` + `tester` + `reviewer` | hierarchical |
| Refactor | `architect` + `coder` + `reviewer` | mesh |
| Security | `security-architect` + `security-auditor` | hierarchical |
| Perf | `perf-analyzer` + `coder` | hierarchical |

## Swarm Topologies

| Topology | Agents | Best For |
|----------|--------|----------|
| `ring` | 1-3 | Sequential pipelines |
| `hierarchical` | 4-6 | Structured, clear authority |
| `mesh` | 4-7 | Collaborative, redundant |
| `hierarchical-mesh` | 7+ | Complex multi-domain |
| `adaptive` | Any | Dynamic workloads |

**Consensus**: `byzantine` (f<n/3), `raft` (f<n/2), `gossip` (eventual), `crdt` (concurrent), `quorum` (tunable)

## Memory

### Syntax (CRITICAL)
```bash
# CORRECT
npx claude-flow@v3alpha memory store "key" '{"data":"value"}' --namespace "project/area"
# WRONG - will fail
npx claude-flow@v3alpha memory store --key "key" --value "value"
```

### Namespaces
```
project/api/[endpoint]    project/events/[type]     project/frontend/[comp]
project/database/[table]  project/tests/[feature]   project/patterns/[type]
```

### HNSW Tuning
```bash
export CLAUDE_FLOW_HNSW_M=16          # Connectivity
export CLAUDE_FLOW_HNSW_EF=200        # Accuracy
export CLAUDE_FLOW_EMBEDDING_DIM=384  # Dimensions
```

## Hooks

| Hook | Purpose |
|------|---------|
| `route "task"` | Route to optimal agent (learned patterns) |
| `pretrain` | Bootstrap intelligence from codebase |
| `pre/post-edit FILE` | Learning around file edits |
| `pre/post-task` | Task lifecycle tracking |
| `session-start/end` | Session management |
| `intelligence pattern-search -q` | HNSW pattern search (150x faster) |
| `worker dispatch --trigger NAME` | Trigger background worker |
| `coverage-gaps` | Test coverage analysis |

## Background Workers (12)

| Worker | Trigger | Auto-Triggers On |
|--------|---------|------------------|
| `map` | 5min | New dirs, large changes |
| `audit` | 10min | Security file changes |
| `optimize` | 15min | Slow ops detected |
| `testgaps` | 20min | Code without tests |
| `consolidate` | 30min | Session end |
| `ultralearn` | manual | New project |
| `deepdive` | manual | Complex edits |
| `refactor` | manual | Code smells |
| `benchmark` | manual | Perf-critical changes |

## Optimal Patterns

### Auto-Swarm Trigger (complex tasks)
- 3+ files modified
- New feature/refactor
- API changes with tests
- Security/performance work

### Sequential Workflow Template
```javascript
// Message 1: Agent 1/N
Task("backend-dev", `
## TASK: [description]
## CONTEXT: Agent #1/4 | Next: [agents + what they need]
## STORAGE: "project/[ns]/[key]" - [data for next agents]
`)

// Message 2: Agent 2/N (WAIT)
Task("coder", `
## TASK: [description]
## CONTEXT: Agent #2/4 | Previous: Backend | Next: [agents]
## RETRIEVAL: npx claude-flow@v3alpha memory retrieve --key "project/[ns]/[key]"
## STORAGE: "project/[ns]/[key]" - [data for next]
`)
// Continue sequentially...
```

### Swarm Auto-Orchestration (ONE message)
```javascript
// All in SAME message:
mcp__claude-flow__swarm_init({topology:"hierarchical-mesh", maxAgents:15, strategy:"adaptive"})

Task("hierarchical-coordinator", "Initialize, coordinate via memory")
Task("researcher", "Analyze requirements, store findings")
Task("system-architect", "Design approach, document decisions")
Task("coder", "Implement following architecture")
Task("tester", "Write tests")
Task("reviewer", "Review quality and security")

TodoWrite({todos: [
  {content:"Init swarm", status:"in_progress", activeForm:"Initializing"},
  {content:"Research", status:"in_progress", activeForm:"Researching"},
  {content:"Design", status:"pending", activeForm:"Designing"},
  {content:"Implement", status:"pending", activeForm:"Implementing"},
  {content:"Test", status:"pending", activeForm:"Testing"},
  {content:"Review", status:"pending", activeForm:"Reviewing"}
]})
```

### Parallel Pattern (read-only ONLY)
```javascript
// ONE message, all parallel
Task("researcher", "Analyze auth patterns")
Task("code-analyzer", "Find vulnerabilities")
Task("perf-analyzer", "Profile endpoints")
```

### Subagent Response Format
```markdown
## TASK COMPLETION
**Did**: [1-2 sentences]
**Files**: `./path/file.ts` - [desc]
**Memory**: `project/ns/key` - [what, why next needs it]
**Retrieve**: `npx claude-flow@v3alpha memory retrieve --key "project/ns/key"`
**Next**: [guidance for downstream agents]
```

## MCP Tools

| Tool | Use |
|------|-----|
| `swarm_init` | `{topology, maxAgents, strategy}` |
| `agent_spawn` | `{type, name, capabilities}` |
| `task_orchestrate` | `{task, strategy, priority}` |
| `memory_usage` | `{action:"store/retrieve/search", namespace, key, value}` |
| `memory_search` | `{pattern, namespace, limit}` |
| `swarm_status` | Health check |
| `agent_metrics` | Performance |
| `neural_train` | Train patterns |
| `performance_report` | `{format, timeframe}` |

## Performance Targets

| Metric | Target |
|--------|--------|
| CLI startup | <500ms |
| MCP response | <100ms |
| Agent spawn | <200ms |
| HNSW search | <1ms |
| SONA adapt | <0.05ms |

## Environment
```bash
CLAUDE_FLOW_MAX_AGENTS=15
CLAUDE_FLOW_TOPOLOGY=hierarchical
CLAUDE_FLOW_MEMORY_BACKEND=hybrid
ANTHROPIC_API_KEY=sk-ant-...
```

## Troubleshooting
```bash
# Health check
npx claude-flow@v3alpha doctor --fix --verbose

# MCP port in use
lsof -i :3000 && kill -9 <PID>

# Empty pattern search
npx claude-flow@v3alpha hooks pretrain

# Memory issues
export CLAUDE_FLOW_MAX_AGENTS=5
npx claude-flow@v3alpha memory cleanup
```

## Checklist

**Before work**: `doctor --fix` → `daemon start` → `mcp start`

**Complex tasks**: Init swarm → Route via hooks → Sequential workflow → Memory handoffs → Batch todos

**After work**: `hooks session-end --export-metrics` → Verify results → Cleanup

---
**Core**: Claude Flow coordinates, Claude Code executes. Sequential default. Forward prompts. Memory handoffs. Truth protocol (no faking).
