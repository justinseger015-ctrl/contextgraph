# MCP Research Findings for ContextGraph Ghost System

## Research Summary

This document contains comprehensive research on the Model Context Protocol (MCP) for implementing the ContextGraph Ghost System MCP stdio server.

---

## 1. What is MCP

The **Model Context Protocol (MCP)** is an open standard introduced by Anthropic in November 2024 to standardize how AI applications connect to external tools, systems, and data sources. Think of MCP as a "USB-C port for LLMs" - it provides a universal interface for:

- Reading files and resources
- Executing functions (tools)
- Handling contextual prompts
- Enabling AI agents to interact with external systems

MCP re-uses the message-flow ideas of the Language Server Protocol (LSP) and is transported over **JSON-RPC 2.0**.

---

## 2. stdio vs SSE Transport

### stdio Transport (Recommended for Local Tools)

| Aspect | Details |
|--------|---------|
| **How it works** | Client launches MCP server as subprocess, communicates via stdin/stdout |
| **Message format** | JSON-RPC 2.0, UTF-8 encoded |
| **Framing** | Newline-delimited (one JSON message per line) |
| **Performance** | Optimal - no network overhead |
| **Security** | Preferred for local tools - no network exposure |
| **Logging** | Use stderr only (never stdout for non-JSON) |

### HTTP/SSE Transport (For Remote Servers)

| Aspect | Details |
|--------|---------|
| **How it works** | HTTP POST for messages, optional SSE for streaming |
| **Use case** | Remote servers, cloud deployments |
| **Authentication** | Supports bearer tokens, API keys, custom headers |

**Why stdio for local tools**: It's the simplest and most secure option. No network socket means no attack surface. The server is isolated as a subprocess.

---

## 3. Protocol Flow

### Complete Initialization Sequence

```
┌─────────────┐                           ┌─────────────┐
│   Client    │                           │   Server    │
│(Claude Code)│                           │ (subprocess)│
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       │  1. Launch server as subprocess         │
       │─────────────────────────────────────────>
       │                                         │
       │  2. initialize request                  │
       │  {protocolVersion, capabilities,        │
       │   clientInfo}                           │
       │─────────────────────────────────────────>
       │                                         │
       │  3. initialize response                 │
       │  {protocolVersion, capabilities,        │
       │   serverInfo}                           │
       <─────────────────────────────────────────│
       │                                         │
       │  4. notifications/initialized           │
       │─────────────────────────────────────────>
       │                                         │
       │  === Normal Operation Phase ===         │
       │                                         │
       │  5. tools/list request                  │
       │─────────────────────────────────────────>
       │                                         │
       │  6. tools array response                │
       <─────────────────────────────────────────│
       │                                         │
       │  7. tools/call request                  │
       │  {name, arguments}                      │
       │─────────────────────────────────────────>
       │                                         │
       │  8. result response                     │
       │  {content, isError}                     │
       <─────────────────────────────────────────│
       │                                         │
```

---

## 4. Message Format (JSON-RPC 2.0)

### Request Structure

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "method/name",
  "params": {
    "key": "value"
  }
}
```

### Response Structure (Success)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "data": "value"
  }
}
```

### Response Structure (Error)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {}
  }
}
```

### Notification Structure (no response expected)

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

---

## 5. Required Methods

### 5.1 initialize (Client -> Server)

**Purpose**: Start handshake, negotiate capabilities and protocol version

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-code",
      "version": "1.0.0"
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": { "listChanged": true }
    },
    "serverInfo": {
      "name": "contextgraph-ghost",
      "version": "1.0.0"
    }
  }
}
```

### 5.2 initialized (Client -> Server, Notification)

**Purpose**: Signal client is ready for normal operations

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### 5.3 tools/list (Client -> Server)

**Purpose**: Discover available tools

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "query_context",
        "description": "Query the context graph for relevant information",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query"
            }
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

### 5.4 tools/call (Client -> Server)

**Purpose**: Invoke a tool

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "query_context",
    "arguments": {
      "query": "authentication patterns"
    }
  }
}
```

**Response (Success)**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 3 relevant context nodes..."
      }
    ],
    "isError": false
  }
}
```

**Response (Tool Error)**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Error: Query too broad, please be more specific"
      }
    ],
    "isError": true
  }
}
```

---

## 6. Claude Code Integration

### Configuration File Locations

| Location | Scope | Purpose |
|----------|-------|---------|
| `.mcp.json` | Project | Shared with team, version-controlled |
| `.claude/settings.local.json` | Project-local | Personal project settings |
| `~/.claude/settings.local.json` | User | Global user settings |

### Configuration Format

```json
{
  "mcpServers": {
    "contextgraph-ghost": {
      "type": "stdio",
      "command": "node",
      "args": ["/absolute/path/to/ghost-mcp-server.js"],
      "env": {
        "CONTEXTGRAPH_DB": "/path/to/context.db"
      }
    }
  }
}
```

### CLI Commands

```bash
# Add server
claude mcp add contextgraph-ghost --scope project

# Add with JSON
claude mcp add-json contextgraph-ghost '{"command":"node","args":["./ghost-mcp-server.js"]}'

# List servers
claude mcp list

# Remove server
claude mcp remove contextgraph-ghost
```

---

## 7. Key Implementation Details

### Critical Requirements

1. **stdout is ONLY for JSON-RPC messages** - Never log to stdout
2. **Use stderr for all logging** - Debugging, errors, status
3. **No embedded newlines** - Each message is one line
4. **UTF-8 encoding required** - All messages must be valid UTF-8
5. **Validate all inputs** - Use JSON Schema validation
6. **Handle unknown methods gracefully** - Return -32601 error

### Server Capabilities Declaration

For a tools-only server:
```json
{
  "capabilities": {
    "tools": {
      "listChanged": false
    }
  }
}
```

### Error Codes Reference

| Code | Name | When to Use |
|------|------|-------------|
| -32700 | Parse error | Invalid JSON received |
| -32600 | Invalid Request | Missing required fields |
| -32601 | Method not found | Unknown method called |
| -32602 | Invalid params | Wrong parameter types |
| -32603 | Internal error | Server-side exceptions |
| -32800 | Request cancelled | Client cancelled request |
| -32801 | Content too large | Response exceeds limits |

### Content Types for Results

```javascript
// Text content
{ "type": "text", "text": "string content" }

// Image content
{ "type": "image", "data": "base64...", "mimeType": "image/png" }

// Resource reference
{ "type": "resource", "resource": { "uri": "file:///...", "text": "..." } }
```

---

## 8. TypeScript Implementation Pattern

### Minimal stdio Server

```typescript
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

// Create server
const server = new McpServer({
  name: 'contextgraph-ghost',
  version: '1.0.0',
});

// Define tool
server.tool(
  'query_context',
  { query: z.string().describe('Search query for context graph') },
  async ({ query }) => {
    // Implementation here
    const results = await searchContextGraph(query);
    return {
      content: [{ type: 'text', text: JSON.stringify(results) }],
    };
  }
);

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // Server now listening on stdin, responding on stdout
}

main().catch(console.error);
```

### Manual Implementation (No SDK)

```typescript
import * as readline from 'readline';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

rl.on('line', (line) => {
  try {
    const request = JSON.parse(line);
    const response = handleRequest(request);
    console.log(JSON.stringify(response));
  } catch (e) {
    console.log(JSON.stringify({
      jsonrpc: '2.0',
      id: null,
      error: { code: -32700, message: 'Parse error' }
    }));
  }
});

function handleRequest(req: any) {
  switch (req.method) {
    case 'initialize':
      return {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          protocolVersion: '2025-03-26',
          capabilities: { tools: {} },
          serverInfo: { name: 'ghost', version: '1.0.0' }
        }
      };
    case 'tools/list':
      return {
        jsonrpc: '2.0',
        id: req.id,
        result: { tools: [...] }
      };
    case 'tools/call':
      return handleToolCall(req);
    default:
      return {
        jsonrpc: '2.0',
        id: req.id,
        error: { code: -32601, message: 'Method not found' }
      };
  }
}
```

---

## Sources

- [Model Context Protocol - Transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [Model Context Protocol - Lifecycle](https://modelcontextprotocol.io/specification/2025-03-26/basic/lifecycle)
- [Model Context Protocol - Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [Model Context Protocol - Prompts](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts)
- [Claude Code MCP Documentation](https://code.claude.com/docs/en/mcp)
- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Error Codes](https://www.mcpevals.io/blog/mcp-error-codes)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/historical/json-rpc-2-0.html)

---

## Memory Location

Research stored at: `contextgraph/ghost-system/mcp-research`

Retrieve with:
```bash
npx claude-flow@alpha memory retrieve "mcp-research" --namespace "contextgraph/ghost-system"
```
