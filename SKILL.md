---
name: mcp-code-rag
category: devops
description: "Use MCP Code RAG server for local code indexing, semantic search, AST analysis, and project mapping. Index any codebase and query it via hybrid search."
---

## ⚡ Quick Start

```bash
# Start the server (transport=stdio for MCP client use)
cd mcp-code-rag
.venv/bin/python -m mcp_code_rag.server --config config.yaml --transport stdio
```

**Hermes Agent configuration** (`~/.hermes/config.yaml`):
```yaml
code-rag:
  command: /path/to/mcp-code-rag/.venv/bin/python
  args: ["-m", "mcp_code_rag.server", "--config", "/path/to/mcp-code-rag/config.yaml", "--transport", "stdio"]
  timeout: 600
  connect_timeout: 120
```

The server uses **transport stdio** by default when launched via Hermes MCP client. No port conflicts. Hermes handles the lifecycle.

## 📋 Project Structure

- venv: `./.venv`
- Index: `./code_rag_index/`
- Config: `./config.yaml`

## 🔧 Prerequisites

- **Python 3.9+**
- **Ollama** running locally (for embeddings + optional LLM tagging)
- **SQLite** (bundled with Python)
- **ChromaDB** (installed via pip, zero Docker required)

### Recommended Ollama Setup

```bash
# Pull embedding model (required)
ollama pull mxbai-embed-large         # Better for code (1.3 GB, 1024 dims)
# or
ollama pull nomic-embed-text          # Default, lightweight

# Pull tagging model (optional, for semantic tagging)
ollama pull qwen3.5:latest
```

## 🎯 17 MCP Tools

### Ingestion & Indexing
1. **`index_project`** — Index a directory
   - `directory`, `force_reindex` (bool), `format` ("human"|"json")
2. **`reindex_file`** — Force re-index a single file
3. **`watch_directory`** — Start/stop file watcher for auto-updates

### Search (Hybrid: BM25 + Vector)
4. **`search_code`** — Hybrid semantic + keyword search
   - `query`, `top_k` (default 10), `workspace_id`, `language_filter`, `exclude_tests`, `format`

### Symbol Navigation
5. **`get_function_details`** — Full function/method source + signature + docstring
6. **`find_type_definition`** — Class/struct/interface definition with all members
7. **`find_implementations`** — Callers (who calls) and targets (what it calls)
8. **`get_code_context`** — What symbol encloses a given line number
9. **`list_package_exports`** — Public symbols in a module/package

### Project Mapping & Analytics
10. **`project_map`** — Structured map (summary/detailed/tree)
11. **`file_tree`** — Directory tree with file statistics
12. **`dependency_graph`** — Import graph with circular deps detection
13. **`project_stats`** — LOC, symbol count, evolution delta

### Diagnostic & Control
14. **`diagnose`** — Health check (ChromaDB, SQLite, Ollama, workspaces)
15. **`get_model_status`** — Ollama model availability
16. **`delete_project`** — Remove indexed workspace
17. **`clear_index`** — Delete ALL data

## 🚀 Recommended Workflow

### Phase 1: Initial Setup
1. Call **`diagnose`** — verify ChromaDB, SQLite, Ollama are operational
2. Call **`get_model_status`** — check embed + tag models available

### Phase 2: Indexing
3. Call **`index_project`** — `directory: "/path/to/project"`, `force_reindex: false`
   - Auto-detects workspace via markers (pyproject.toml, go.mod, package.json, Cargo.toml, .csproj)
   - Computes SHA-256 hashes, skips unchanged files (incremental)
4. Call **`project_map`** with `format: "tree"` — see project structure

### Phase 3: Exploration
5. Call **`project_map`** with `format: "summary"` — get stats, languages, top files
6. Call **`project_stats`** — LOC counts, evolution between scans
7. Call **`file_tree`** — browse with filters (`path_filter: "*.py"`)

### Phase 4: Search & Navigation
8. Call **`search_code`** — natural language or code snippet queries
9. Call **`get_function_details`** — get complete implementation
10. Call **`find_type_definition`** — explore classes/interfaces
11. Call **`find_implementations`** — trace callers and targets
12. Call **`get_code_context`** — see enclosing symbol by line number
13. Call **`list_package_exports`** — see public API of a module

### Phase 5: Dependency Analysis
14. Call **`dependency_graph`** — import structure + circular deps
15. Call **`project_stats`** — track evolution

### Phase 6: Real-Time Monitoring (Optional)
16. Call **`watch_directory`** — auto-update on file changes
17. Call **`reindex_file`** — force re-index single file

## ⚠️ Pitfalls & Known Issues

### Workspace auto-detection fails for some C# projects
The detector looks for `.csproj` at the root level. Projects with `.sln` at root and `.csproj` in subdirs — detection fails.

**Fix:** Manually pass `WorkspaceInfo` to `ingest_directory`:
```python
from mcp_code_rag.workspace import WorkspaceInfo
ws = WorkspaceInfo(
    id="project-name",
    name="project-name",
    root="/path/to/project",
    marker_type=".sln",
    languages=["csharp"]
)
pipeline.ingest_directory('/path/to/project', workspace=ws)
```

### Embedding errors (400 Bad Request from Ollama)
Some chunks (minified JSON, build artifacts) exceed the embedding model's token limit. FTS and symbols still work — only vector scores are affected for failed chunks.

### map_generator bugs
Three bugs were found and patched in `map_generator.py`:
1. Singular/plural mismatch: `function` vs `functions` in DB→dict mapping
2. Symbol lookup used filename-only instead of absolute path
3. `_render_tree` wrong key for `symbol_counts` lookup

## 🔧 Configuration (config.yaml)

Key settings:
```yaml
ollama:
  base_url: "http://localhost:11434"
  embed_model: "mxbai-embed-large"    # 1024 dims
  tag_model: "qwen3.5:latest"
  auto_pull: true

rag:
  index_path: "./code_rag_index"
  chunk_strategy: "ast"
  max_embed_text_length: 8000

hybrid_search:
  enabled: true
  alpha: 0.6
  beta: 0.4
```
