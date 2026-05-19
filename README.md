# MCP Code RAG — Code Search & Analysis Server

A specialized MCP (Model Context Protocol) server for **Hermes Agent** providing **local code search**, **semantic analysis**, **AST-aware chunking**, **hybrid search**, and **real-time file watching**. Index any codebase locally and query it via 17 semantic and structural tools.

## Features

- **Hybrid Search**: Combines BM25 keyword + semantic vector search with intelligent reranking
- **Multi-Language AST Support**: Python (full AST), JavaScript/TypeScript/Go/C#/Rust (regex patterns), + 15 more languages
- **Workspace Auto-Detection**: Automatically detects Python, Go, Node.js, .NET, Rust, Java, PHP, Ruby projects
- **Incremental Indexing**: Hash cache prevents re-processing unchanged files (5 files changed = 5 files re-indexed)
- **Real-Time Watcher**: Automatic index updates on file changes with debounce and intelligent upsert
- **Structured Output**: JSON descriptors for functions, classes, imports, and dependency graphs
- **Tagging System**: Automatic language, framework, layer, and design pattern detection

## Prerequisites

- **Python 3.9+**
- **Ollama** running locally (for embeddings + optional LLM tagging)
- **SQLite** (bundled with Python)
- **ChromaDB** (installed via pip, zero Docker required)

### Recommended Ollama Setup

```bash
# Pull embedding model (required)
ollama pull nomic-embed-text          # Default, lightweight
# or
ollama pull mxbai-embed-large         # Better for code (1.3 GB)

# Pull tagging model (optional, for semantic tagging)
ollama pull qwen2:1.5b                # Fast, good for H2 tagging
```

Ollama is configured to listen on `http://172.28.128.1:11434` (or use `http://localhost:11434` for direct TCP).

## Installation

### Option 1: Direct Python (Development)

```bash
# Clone and install
git clone <repository>
cd mcp-code-rag
pip install -e .

# Copy and adjust config
cp config.example.yaml config.yaml
# Edit config.yaml: adjust ollama.base_url, rag.index_path, etc.
```

### Option 2: Docker Compose (Recommended)

```bash
# Start Ollama + MCP Code RAG
docker-compose up -d

# Check services
docker-compose logs -f mcp-code-rag
```

Automatically pulls embeddings model and starts the server on `http://localhost:3005/mcp`.

## Configuration

### config.yaml Structure

```yaml
ollama:
  base_url: "http://172.28.128.1:11434"      # Ollama API endpoint
  embed_model: "nomic-embed-text"             # Embedding model
  tag_model: "qwen2:1.5b"                     # Tagging model (LLM)
  timeout_s: 30.0
  embed_timeout_s: 120.0
  max_retries: 3
  auto_pull: false

rag:
  index_path: "./code_rag_index"              # ChromaDB + SQLite location
  chunk_strategy: "ast"                       # "ast" | "regex" | "text"
  max_embed_text_length: 8000                 # Max chars for embedding
  supported_extensions: [.py, .js, .ts, ...]  # Languages to index

hybrid_search:
  enabled: true
  alpha: 0.6                                  # 60% semantic, 40% keyword
  beta: 0.4
  fts5_table: "code_fts"

tagging:
  auto_tag_enabled: true                      # H1 heuristics (instant)
  h2_enabled: false                           # H2 LLM (slower, optional)
  use_cache: true
  taxonomy: {...}                             # Tag categories

watcher:
  enabled: false                              # Enable file monitoring
  debounce_ms: 3000
  sync_deletions: true

workspace:
  auto_detect: true                           # Detect via markers
  cache_enabled: true

logging:
  level: "INFO"
  format: "text" | "json"
  file: ""                                    # "" = stdout only
```

See `config.example.yaml` for full reference with all parameters explained.

## Quick Start

### 1. Start the Server

**Direct:**
```bash
python -m mcp_code_rag.server --config config.yaml --port 3005
```

**Docker:**
```bash
docker-compose up -d
```

### 2. Index a Project

```bash
# Via Hermes Agent MCP call:
# tool: index_project
# arguments: 
#   directory: "/path/to/project"
#   force_reindex: false
#   format: "human"

# Via curl (if you prefer):
curl -X POST http://localhost:3005/mcp \
  -H "Content-Type: application/json" \
  -d '{"method": "index_project", "params": {"directory": "/path/to/project"}}'
```

### 3. Search Code

```
tool: search_code
arguments:
  query: "password validation"
  top_k: 10
  workspace_id: "default"
  format: "human"
```

### 4. Explore Project Map

```
tool: project_map
arguments:
  workspace_id: "default"
  map_format: "tree"
  output_format: "human"
```

---

## 17 MCP Tools Reference

### **Ingestion & Indexing**

#### 1. `index_project` — Index a directory
Scans a directory, detects workspace, chunks code by AST/regex, embeds, and stores.

**Arguments:**
- `directory` (str): Path to index
- `force_reindex` (bool, default=False): Ignore hash cache, reindex all files
- `format` (str, default="human"): "human" or "json"

**Example:**
```
tool: index_project
params:
  directory: "/home/user/projects/review-analyzer"
  force_reindex: false
  format: "human"
```

**Output (human):**
```
Indexed 42 files (243 chunks) in 12450ms.
Languages: python=28, typescript=12, config=2
Workspace: review-analyzer (auto-detected from pyproject.toml)
```

---

#### 2. `reindex_file` — Force re-index a single file
Upserts a single file regardless of hash cache.

**Arguments:**
- `file_path` (str): Absolute or relative path to the file
- `workspace_id` (str, default="default"): Workspace to index into
- `format` (str, default="human"): Output format ("human" or "json")

**Example:**
```
tool: reindex_file
params:
  file_path: "/home/user/projects/review-analyzer/mcp_rag/server.py"
  workspace_id: "review-analyzer"
  format: "json"
```

---

#### 3. `watch_directory` — Real-time file watcher
Start monitoring a directory for changes. On create/modify/delete, automatically re-indexes with intelligent debounce and upsert.

**Arguments:**
- `directory` (str): Path to watch
- `workspace_id` (str, default="default"): Workspace identifier
- `recursive` (bool, default=True): Watch subdirectories
- `enabled` (bool, default=True): True = start, False = stop watching
- `format` (str, default="human"): Output format ("human" or "json")

**Example (start):**
```
tool: watch_directory
params:
  directory: "/home/user/projects/review-analyzer"
  workspace_id: "review-analyzer"
  recursive: true
  enabled: true
  format: "human"
```

**Example (stop):**
```
tool: watch_directory
params:
  enabled: false
  workspace_id: "review-analyzer"
  format: "human"
```

---

### **Search (Hybrid: BM25 + Vector)**

#### 4. `search_code` — Hybrid code search
BM25 keyword search + semantic vector search with reranking.

**Arguments:**
- `query` (str): Natural language or code snippet
- `top_k` (int, default=10): Max results
- `workspace_id` (str, default="default"): Search scope
- `language_filter` (str, optional): Filter by language (e.g., "python")
- `file_filter` (str, optional): Filter by file path
- `exclude_tests` (bool, default=False): Skip test files
- `format` (str, default="human"): Output format ("human" or "json")

**Example:**
```
tool: search_code
params:
  query: "how to verify password against bcrypt hash"
  top_k: 5
  workspace_id: "review-analyzer"
  language_filter: "python"
  exclude_tests: true
  format: "json"
```

**Output (json):**
```json
{
  "results": [
    {
      "type": "method",
      "name": "verify_password",
      "language": "python",
      "signature": "async def verify_password(self, password: str) -> bool",
      "docstring": "Verify password against stored hash using bcrypt.",
      "code": "async def verify_password(self, password: str) -> bool:\n    return bcrypt.verify(...)",
      "location": {"file_path": "auth.py", "start_line": 42, "end_line": 48},
      "hybrid_score": 0.87,
      "tags": ["lang:python", "layer:service"]
    }
  ],
  "query": "verify password bcrypt"
}
```

---

### **Symbol Navigation**

#### 5. `get_function_details` — Full function/method details
Retrieve complete FunctionDescriptor (signature, docstring, code, location).

**Arguments:**
- `symbol_name` (str): Function or method name
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): "human" or "json"

**Example:**
```
tool: get_function_details
params:
  symbol_name: "verify_password"
  workspace_id: "review-analyzer"
  output_format: "json"
```

**Output (json):**
```json
{
  "language": "python",
  "kind": "method",
  "name": "verify_password",
  "namespace": "AuthService",
  "signature": "async def verify_password(self, password: str) -> bool",
  "description": "Verify password against stored hash using bcrypt.",
  "location": {
    "file_path": "/home/.../auth.py",
    "start_line": 42,
    "end_line": 48
  },
  "code": "async def verify_password(self, password: str) -> bool:\n    return bcrypt.verify(password, self.hashed_password)"
}
```

---

#### 6. `find_type_definition` — Class/interface/struct definition
Retrieve ClassDescriptor with fields, methods, base classes.

**Arguments:**
- `type_name` (str): Class, interface, or struct name
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: find_type_definition
params:
  type_name: "AuthService"
  workspace_id: "review-analyzer"
  output_format: "human"
```

**Output (human):**
```
Type:     AuthService
Kind:     class
Language: python
Namespace: mcp_rag.auth
Location: mcp_rag/auth.py:10-120

Methods (5):
  - verify_password  (mcp_rag/auth.py:42-48)
  - hash_password    (mcp_rag/auth.py:50-55)
  - create_token     (mcp_rag/auth.py:57-70)
```

---

#### 7. `find_implementations` — Callers and targets
Find what calls a symbol (callers) or what a symbol imports (targets).

**Arguments:**
- `symbol_name` (str): Symbol to trace
- `direction` (str, default="callers"): "callers" (who calls) | "targets" (what it calls)
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: find_implementations
params:
  symbol_name: "verify_password"
  direction: "callers"
  workspace_id: "review-analyzer"
  format: "human"
```

**Output:**
```
7 callers of 'verify_password':
  - authenticate  (api/routes.py)
  - process_login  (services/auth.py)
  - validate_session  (middleware.py)
```

---

#### 8. `get_code_context` — Code around a line
Find which function/class encloses a given line.

**Arguments:**
- `file_path` (str): Absolute path to file
- `line` (int): Line number (1-indexed)
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: get_code_context
params:
  file_path: "/home/user/projects/review-analyzer/auth.py"
  line: 45
  workspace_id: "review-analyzer"
  format: "human"
```

**Output:**
```
Line 45 is inside method `verify_password`
  (auth.py:42-48)

Enclosing context:
  - class `AuthService` (auth.py:10-120)
  - module `mcp_rag.auth` (auth.py:1-200)
```

---

### **Project Mapping & Analytics**

#### 9. `project_map` — Structured project overview
Generate a map showing files, languages, symbols, dependencies.

**Arguments:**
- `workspace_id` (str, default="default"): Workspace to map
- `map_format` (str, default="summary"): "summary" | "detailed" | "tree" (maps project structure)
- `output_format` (str, default="human"): Output format ("human" or "json")

**Example:**
```
tool: project_map
params:
  workspace_id: "review-analyzer"
  map_format: "tree"
  output_format: "human"
```

**Output (human/tree):**
```
Project Map  [review-analyzer]
Files:  42
Lines:  9182

Languages:
  python: 28 files
  typescript: 12 files
  config: 2 files

Top-level folders: mcp_rag, tests, scripts, config

File Tree:
src/
├── main.py                      [42 LOC, 3 funcs, 1 class]
├── models/
│   ├── user.py                  [128 LOC, 8 funcs, 2 classes]
│   └── review.py                [95 LOC, 6 funcs, 2 classes]
├── api/
│   ├── router.py                [56 LOC, 12 endpoints]
│   └── middleware.py            [34 LOC, 2 funcs]
└── tests/
```

---

#### 10. `file_tree` — Annotated file listing
Display directory tree with file statistics.

**Arguments:**
- `workspace_id` (str, default="default"): Workspace
- `path_filter` (str, optional): Glob filter (e.g., "*.py")
- `max_depth` (int, default=10): Max directory depth
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: file_tree
params:
  workspace_id: "review-analyzer"
  path_filter: "*.py"
  max_depth: 3
  format: "human"
```

---

#### 11. `list_package_exports` — Public symbols in a package
List all non-private (non-underscore) symbols exported by a module/package.

**Arguments:**
- `package_path` (str): Package path (e.g., "mcp_code_rag", "src/models")
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: list_package_exports
params:
  package_path: "mcp_code_rag.auth"
  workspace_id: "review-analyzer"
  format: "json"
```

**Output:**
```json
{
  "package": "mcp_code_rag.auth",
  "exports": [
    "AuthService",
    "verify_password",
    "create_token",
    "TokenData"
  ]
}
```

---

#### 12. `dependency_graph` — Import graph analysis
Build dependency graph showing which files/symbols import which.

**Arguments:**
- `workspace_id` (str, default="default"): Workspace
- `entry_point` (str, optional): Root symbol for BFS subgraph
- `max_depth` (int, optional): Max BFS depth
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: dependency_graph
params:
  workspace_id: "review-analyzer"
  entry_point: null
  output_format: "json"
```

**Output (edges):**
```json
{
  "edges": [
    {
      "source": "api/router.py",
      "target": "services/review.py",
      "import_symbols": ["process_review"],
      "import_type": "direct"
    }
  ],
  "total_files": 28,
  "total_edges": 47,
  "most_imported": [
    {"path": "models/review.py", "import_count": 12}
  ],
  "circular_dependencies": [
    ["models/user.py", "services/auth.py", "models/user.py"]
  ]
}
```

---

#### 13. `project_stats` — Statistics with evolution delta
Get current stats and compare vs. previous scan.

**Arguments:**
- `workspace_id` (str, default="default"): Workspace
- `output_format` (str, default="human"): Output format

**Example:**
```
tool: project_stats
params:
  workspace_id: "review-analyzer"
  format: "json"
```

**Output:**
```json
{
  "current": {
    "files": 42,
    "total_lines": 9182,
    "total_symbols": 243
  },
  "delta": {
    "files_delta": 2,
    "chunks_delta": 15,
    "time_span_ms": 86400000
  },
  "scan_history": [...]
}
```

---

### **Diagnostic & Control**

#### 14. `diagnose` — Health check
Verify ChromaDB, SQLite, Ollama, and list all workspaces.

**Arguments:** (none)

**Example:**
```
tool: diagnose
params: {}
```

**Output:**
```json
{
  "chromadb": {
    "status": "ok",
    "collections": 3,
    "path": "/app/code_rag_index/chroma"
  },
  "sqlite": {
    "status": "ok",
    "path": "/app/code_rag.db",
    "indexed_files": 42
  },
  "ollama": {
    "status": "ok",
    "models_available": 5,
    "embed_model": "nomic-embed-text",
    "embed_model_available": true
  },
  "workspaces": [
    {"id": "a1b2c3", "name": "review-analyzer", "root": "/home/.../review-analyzer"}
  ]
}
```

---

#### 15. `get_model_status` — Ollama model availability
List Ollama models and report if embed/tag models are available.

**Arguments:** (none)

**Example:**
```
tool: get_model_status
params: {}
```

**Output:**
```json
{
  "status": "ok",
  "models": [...],
  "embed_model": {
    "name": "nomic-embed-text",
    "available": true
  },
  "tag_model": {
    "name": "qwen2:1.5b",
    "available": true
  }
}
```

---

#### 16. `delete_project` — Remove indexed workspace
Delete all indexed data for a workspace.

**Arguments:**
- `workspace_id` (str): The 12-character workspace ID to delete

**Example:**
```
tool: delete_project
params:
  workspace_id: "a1b2c3"
```

---

#### 17. `clear_index` — Delete ALL data
Nuke all workspaces and collections.

**Arguments:** (none)

**Example:**
```
tool: clear_index
params: {}
```

---

## Architecture (ASCII Diagram)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Hermes Agent                                │
│            (MCP Client → StreamableHTTP :3005)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   server.py     │  FastMCP
                    │ 17 MCP Tools    │  (index, search, map, watch)
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ CodeIngestPipeline                │  │  HybridSearch  │
│ ├─ detect_workspace               │  │  ├─ BM25 (FTS5)│
│ ├─ detect_language                │  │  ├─ Vector     │
│ ├─ extract_structure              │  │  └─ Reranking  │
│ ├─ chunk_ast() / regex()          │  │                │
│ ├─ tag (H1 + H2)                  │  └────────────────┘
│ ├─ embed via Ollama               │
│ └─ store()                        │
└───────┬────────┘                  │
        │                           │
        └──────────┬────────────────┘
                   │
        ┌──────────▼──────────┐
        │      Storage        │
        │                     │
        ├─ ChromaDB          │  Collections per workspace×lang
        │  (code_rag_index)   │  coderag-{id}-python
        │                     │  coderag-{id}-typescript
        ├─ SQLite            │  code_rag.db
        │  ├─ path_index     │  file path → doc_id
        │  ├─ symbol_index   │  symbol lookup
        │  ├─ dependency_idx │  import graph
        │  ├─ hash_cache     │  SHA256 for incremental
        │  ├─ workspace_cache│  detected workspaces
        │  ├─ scan_history   │  evolution tracking
        │  └─ code_fts       │  full-text search
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Ollama Client     │
        │  embeddings + LLM  │
        │  (optional tagging)│
        └────────────────────┘

┌────────────────────────────────────┐
│  FileWatcher (watch_directory)     │
│  ├─ Debounce (3000ms)              │
│  ├─ Hash check for changes         │
│  ├─ Upsert on modify/create        │
│  └─ Delete from index on removal   │
└────────────────────────────────────┘
```

---

## Recommended Execution Order

### Phase 1: Initial Setup
1. **diagnose** — Verify ChromaDB, SQLite, Ollama are running
2. **get_model_status** — Check embed + tag models available

### Phase 2: Indexing
3. **index_project** — Scan and index directory with AST/regex/text chunking
   - Auto-detects workspace (py, go, npm, etc.)
   - Computes hashes, skips unchanged files
   - Embeds code via Ollama
   - Stores in ChromaDB + SQLite

### Phase 3: Exploration
4. **project_map** (format="tree") — Get project structure overview
5. **project_stats** — See lines of code, symbol count, evolution
6. **file_tree** — Browse files with filters

### Phase 4: Search & Navigation
7. **search_code** — Find code by natural language or patterns
8. **get_function_details** — Get complete function/method
9. **find_type_definition** — Explore classes, interfaces, structs
10. **find_implementations** — Trace callers/targets
11. **get_code_context** — See what symbol encloses a line
12. **list_package_exports** — See public API of a module

### Phase 5: Dependency Analysis
13. **dependency_graph** — Visualize import structure + circular deps
14. **project_stats** — Track evolution between scans

### Phase 6: Real-Time Monitoring (Optional)
15. **watch_directory** — Auto-update index on file changes
16. **reindex_file** — Force re-index a single file

### Phase 7: Cleanup
17. **delete_project** — Remove workspace (keep others)
18. **clear_index** — Nuke entire index

---

## Hermes Agent Configuration

Add to Hermes Agent's `.config/hermes/mcp_servers.yaml`:

```yaml
mcp_servers:
  code-rag:
    url: http://localhost:3005/mcp
    timeout: 600
    connect_timeout: 120
```

This enables Hermes to:
- Index any project directory
- Search code semantically
- Navigate functions, classes, imports
- Analyze dependencies
- Track project evolution
- Auto-update on file changes

---

## Performance Notes

| Operation | Typical Time |
|-----------|---|
| **index_project** (1000 files, 500K LOC) | 45–60s |
| **search_code** (top-10 results) | 200–500ms |
| **get_function_details** | <50ms |
| **dependency_graph** | 500ms–2s (depending on graph size) |
| **watch_directory** (on file change) | 1–3s (includes debounce) |
| **project_stats** | <100ms |

**Scaling:**
- **Hash cache** eliminates re-processing: 1000 files → 5 changed = 5 re-indexed
- **Symbol index** enables instant lookups (SQLite indexed)
- **FTS5** accelerates keyword search across 10K+ symbols
- **ChromaDB** batches vector operations efficiently

---

## Troubleshooting

### "Ollama connection refused"
```bash
# Check Ollama is running
curl http://172.28.128.1:11434/api/tags
# or locally:
curl http://localhost:11434/api/tags

# In config.yaml, verify:
ollama:
  base_url: "http://172.28.128.1:11434"  # or http://localhost:11434
```

### "Embed model not available"
```bash
# Pull the model
ollama pull nomic-embed-text

# Or configure a different model:
# config.yaml → ollama.embed_model = "mxbai-embed-large"
```

### "ChromaDB not initializing"
Check permissions on `rag.index_path` in config.yaml:
```bash
ls -la ./code_rag_index
# Should be writable by the process
```

### "Workspace not detected"
If auto-detection fails, manually specify:
```
tool: index_project
params:
  directory: "/path/to/project"
  workspace: "my-project"  # optional override
```

---

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=mcp_code_rag
```

Key test files:
- `test_ast_chunker.py` — Python AST parsing
- `test_regex_chunker.py` — JavaScript/Go/C# patterns
- `test_hybrid_search.py` — BM25 + vector reranking
- `test_ingestion.py` — Pipeline end-to-end
- `test_storage.py` — ChromaDB + SQLite operations
- `test_workspace.py` — Auto-detection + caching
- `test_dependency_parser.py` — Import graph analysis

---

## Project Structure

```
mcp-code-rag/
├── pyproject.toml                 # Dependencies, CLI entrypoint
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml             # Ollama + MCP Code RAG
├── docker-entrypoint.sh           # Docker startup script
├── config.example.yaml            # Configuration template
├── README.md                      # This file
├── docs/
│   ├── spec.md                   # Full architecture spec
│   ├── incremental_indexing.md   # Hash cache algorithm
│   ├── AVANCEMENT.md             # Phase tracking
│   └── CODING_PLAN.md            # Design decisions
├── mcp_code_rag/
│   ├── __init__.py
│   ├── server.py                 # FastMCP server (17 tools)
│   ├── config.py                 # Pydantic settings
│   ├── ingest.py                 # CodeIngestPipeline
│   ├── chunker.py                # AST + regex chunkers
│   ├── extractors.py             # Language detection, symbol extraction
│   ├── storage.py                # ChromaDB + SQLite
│   ├── hybrid_search.py          # BM25 + vector reranking
│   ├── workspace.py              # Auto-detection + markers
│   ├── map_generator.py          # Project map, stats, trees
│   ├── dependency.py             # Import graph analysis
│   ├── tagging/
│   │   ├── __init__.py
│   │   ├── heuristics.py         # H1 tagging (instant)
│   │   └── llm_tagger.py         # H2 LLM tagging (optional)
│   ├── watcher/
│   │   ├── __init__.py
│   │   └── fs_watcher.py         # watchdog + debounce
│   ├── utils/
│   │   ├── __init__.py
│   │   └── hashing.py            # SHA-256 file hashing
│   ├── ollama_client.py          # HTTP client for Ollama
│   └── logging_config.py         # Structured logging
├── tests/
│   ├── test_ast_chunker.py
│   ├── test_regex_chunker.py
│   ├── test_codechunk.py
│   ├── test_hybrid_search.py
│   ├── test_ingestion.py
│   ├── test_workspace.py
│   ├── test_storage.py
│   ├── test_tagging_heuristics.py
│   ├── test_dependency_parser.py
│   ├── test_map_generator.py
│   └── test_server_phase2.py
└── code_rag_index/               # Index directory (gitignored)
    ├── chroma/                   # ChromaDB persistent storage
    └── code_rag.db               # SQLite database
```

---

## License

MIT

## Authors

MCP Code RAG Contributors
