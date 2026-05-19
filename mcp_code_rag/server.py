"""FastMCP server exposing 15 MCP tools for code indexing and search."""

import json
import logging
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fastmcp
from fastmcp import FastMCP

from mcp_code_rag.config import AppConfig, load_config
from mcp_code_rag.dependency import DependencyAnalyzer
from mcp_code_rag.extractors import (
    ClassDescriptor,
    FunctionDescriptor,
    SymbolLocation,
    detect_language,
)
from mcp_code_rag.hybrid_search import HybridSearch
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.logging_config import setup_logging
from mcp_code_rag.map_generator import MapGenerator
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.storage import Storage
from mcp_code_rag.watcher.fs_watcher import FileWatcher
from mcp_code_rag.workspace import list_workspaces

logger = logging.getLogger(__name__)


def _get_chunk_meta(
    storage: Storage, workspace_id: str, file_path: str, start_line: int, end_line: int
) -> dict:
    """Retrieve chunk metadata dict from ChromaDB by chunk ID."""
    chunk_id = f"{workspace_id}_{file_path}_{start_line}_{end_line}"
    try:
        for coll_info in storage.chroma_client.list_collections():
            if workspace_id in coll_info.name:
                coll = storage.chroma_client.get_collection(coll_info.name)
                result = coll.get(ids=[chunk_id], include=["metadatas"])
                if result["ids"]:
                    return result["metadatas"][0]
    except Exception:
        pass
    return {}


def create_server(
    config: AppConfig,
    storage: Storage,
    ollama_client: OllamaClient,
    pipeline: CodeIngestPipeline,
    search: HybridSearch,
) -> FastMCP:
    """Create and configure the FastMCP server with all tools."""
    mcp = FastMCP("mcp-code-rag")

    # Initialize FileWatcher for watch_directory tool
    file_watcher = FileWatcher(pipeline, storage, config)

    @mcp.tool()
    def index_project(
        directory: str,
        force_reindex: bool = False,
        format: str = "human",
    ) -> str:
        """Index a project directory into the code search index.

        Args:
            directory: Absolute or relative path to the project directory.
            force_reindex: Re-index all files even if unchanged.
            format: Output format — "human" (default) or "json".
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        result = pipeline.ingest_directory(str(dir_path), force_reindex=force_reindex)

        if format == "json":
            return json.dumps(result, indent=2)

        files_indexed = result["files_indexed"]
        total_chunks = result["total_chunks"]
        duration_ms = result["duration_ms"]
        languages = result.get("languages", {})
        errors = result.get("errors", [])

        lines = [f"Indexed {files_indexed} files ({total_chunks} chunks) in {duration_ms}ms."]
        if languages:
            lang_str = ", ".join(f"{lang}={count}" for lang, count in languages.items())
            lines.append(f"Languages: {lang_str}")
        if errors:
            lines.append(f"Errors ({len(errors)}):")
            for err in errors[:5]:
                lines.append(f"  - {err}")
        return "\n".join(lines)

    @mcp.tool()
    def search_code(
        query: str,
        top_k: int = 10,
        workspace_id: str = "default",
        language_filter: Optional[str] = None,
        file_filter: Optional[str] = None,
        exclude_tests: bool = False,
        format: str = "human",
    ) -> str:
        """Search indexed code using hybrid BM25 + semantic vector search.

        Args:
            query: Natural language or code search query.
            top_k: Maximum number of results to return (default 10).
            workspace_id: Workspace to search (default "default").
            language_filter: Filter by language, e.g. "python", "go".
            file_filter: Filter by file path substring.
            exclude_tests: Exclude files with "test" in their path.
            format: Output format — "human" (default) or "json".
        """
        results = search.hybrid_search(
            query=query,
            top_k=top_k,
            workspace_id=workspace_id,
            language_filter=language_filter,
            file_filter=file_filter,
            exclude_tests=exclude_tests,
        )

        if format == "json":
            return json.dumps(results, indent=2)

        if not results:
            return f"No results found for '{query}'."

        lines = [f"Found {len(results)} result(s) for '{query}':\n"]
        for i, r in enumerate(results, 1):
            score = r.get("hybrid_score", 0.0)
            symbol = r.get("symbol_name", "")
            file_path = r.get("file_path", "")
            code = (r.get("code", "") or "")[:120]
            lines.append(f"{i}. `{symbol}` in {file_path} (score: {score:.3f})")
            if code:
                lines.append(f"   {code}")
            lines.append("")
        return "\n".join(lines)

    @mcp.tool()
    def delete_project(workspace_id: str) -> str:
        """Delete all indexed data for a workspace.

        Args:
            workspace_id: The 12-character workspace ID to delete.
        """
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM path_index WHERE workspace_id = ?", (workspace_id,)
        )
        files_count = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM symbol_index WHERE workspace_id = ?", (workspace_id,)
        )
        symbols_count = cursor.fetchone()[0]
        conn.close()

        try:
            collections = storage.chroma_client.list_collections()
            has_collections = any(workspace_id in c.name for c in collections)
        except Exception:
            has_collections = False

        if files_count == 0 and symbols_count == 0 and not has_collections:
            raise ValueError(f"Workspace not found: {workspace_id}")

        storage.delete_workspace(workspace_id)
        return f"Deleted workspace '{workspace_id}' successfully."

    @mcp.tool()
    def clear_index() -> str:
        """Delete ALL indexed data from every workspace."""
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT workspace_id FROM path_index")
        workspace_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        for ws_id in workspace_ids:
            storage.delete_workspace(ws_id)

        # Clear any orphan ChromaDB collections
        try:
            collections = storage.chroma_client.list_collections()
            for coll in collections:
                storage.chroma_client.delete_collection(coll.name)
        except Exception as e:
            logger.warning("Could not clear ChromaDB collections: %s", e)

        # Truncate all remaining SQLite tables
        conn = sqlite3.connect(storage.db_path)
        for table in [
            "path_index",
            "symbol_index",
            "dependency_index",
            "scan_history",
            "hash_cache",
            "workspace_cache",
        ]:
            conn.execute(f"DELETE FROM {table}")  # noqa: S608 – names are hardcoded
        conn.execute("DELETE FROM code_fts")
        conn.commit()
        conn.close()

        count = len(workspace_ids)
        return f"Cleared {count} workspace(s) from the index."

    @mcp.tool()
    def diagnose() -> str:
        """Check the health of ChromaDB, SQLite, Ollama, and list workspaces."""
        report: dict = {}

        # ChromaDB
        try:
            collections = storage.chroma_client.list_collections()
            report["chromadb"] = {
                "status": "ok",
                "collections": len(collections),
                "path": str(storage.index_path / "chroma"),
            }
        except Exception as e:
            report["chromadb"] = {"status": "error", "error": str(e)}

        # SQLite
        try:
            conn = sqlite3.connect(storage.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM path_index")
            indexed_files = cursor.fetchone()[0]
            conn.close()
            report["sqlite"] = {
                "status": "ok",
                "path": str(storage.db_path),
                "indexed_files": indexed_files,
            }
        except Exception as e:
            report["sqlite"] = {"status": "error", "error": str(e)}

        # Ollama
        try:
            models = ollama_client.list_models()
            model_names = [m.get("name", "") for m in models]
            report["ollama"] = {
                "status": "ok",
                "models_available": len(models),
                "embed_model": config.ollama.embed_model,
                "embed_model_available": any(
                    config.ollama.embed_model in name for name in model_names
                ),
            }
        except Exception as e:
            report["ollama"] = {"status": "error", "error": str(e)}

        # Workspaces
        try:
            workspaces = list_workspaces(str(storage.db_path))
            report["workspaces"] = [
                {"id": w.id, "name": w.name, "root": w.root} for w in workspaces
            ]
        except Exception as e:
            report["workspaces"] = {"error": str(e)}

        return json.dumps(report, indent=2)

    @mcp.tool()
    def get_model_status() -> str:
        """List Ollama models and report availability of embed and tag models."""
        try:
            models = ollama_client.list_models()
            model_names = [m.get("name", "") for m in models]
            return json.dumps(
                {
                    "status": "ok",
                    "models": models,
                    "embed_model": {
                        "name": config.ollama.embed_model,
                        "available": any(
                            config.ollama.embed_model in name for name in model_names
                        ),
                    },
                    "tag_model": {
                        "name": config.ollama.tag_model,
                        "available": any(
                            config.ollama.tag_model in name for name in model_names
                        ),
                    },
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    # ── Phase 2 tools ────────────────────────────────────────────────────────

    @mcp.tool()
    def get_function_details(
        symbol_name: str,
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """Get detailed information about a function or method by name.

        Looks up the symbol in the index, retrieves its code from ChromaDB,
        and returns a structured FunctionDescriptor.

        Args:
            symbol_name: The function or method name to look up.
            workspace_id: Workspace to search (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        conn = sqlite3.connect(storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT symbol_name, symbol_type, file_path, start_line, end_line
            FROM symbol_index
            WHERE workspace_id = ? AND symbol_name = ?
              AND symbol_type IN ('function', 'method')
            LIMIT 1
            """,
            (workspace_id, symbol_name),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(
                f"Function/method '{symbol_name}' not found in workspace '{workspace_id}'"
            )

        file_path = row["file_path"]
        start_line = row["start_line"]
        end_line = row["end_line"]
        symbol_type = row["symbol_type"]

        cursor.execute(
            """
            SELECT code, docstring FROM code_fts
            WHERE workspace_id = ? AND file_path = ? AND symbol_name = ?
            LIMIT 1
            """,
            (workspace_id, file_path, symbol_name),
        )
        fts_row = cursor.fetchone()
        conn.close()

        code = fts_row["code"] if fts_row else ""
        docstring = fts_row["docstring"] if fts_row else ""

        meta = _get_chunk_meta(storage, workspace_id, file_path, start_line, end_line)
        signature = meta.get("signature", "")
        language = meta.get("language") or detect_language(file_path) or "unknown"

        descriptor = FunctionDescriptor(
            language=language,
            kind=symbol_type,
            name=symbol_name,
            namespace=meta.get("package") or None,
            signature=signature,
            description=docstring or None,
            location=SymbolLocation(
                file_path=file_path, start_line=start_line, end_line=end_line
            ),
            code=code or None,
        )

        if output_format == "json":
            return json.dumps(asdict(descriptor), indent=2)

        lines = [
            f"Function: {descriptor.name}",
            f"Kind:     {descriptor.kind}",
            f"Language: {descriptor.language}",
        ]
        if descriptor.namespace:
            lines.append(f"Namespace: {descriptor.namespace}")
        if descriptor.signature:
            lines.append(f"Signature: {descriptor.signature}")
        if descriptor.description:
            lines.append(f"Description: {descriptor.description}")
        lines.append(f"Location: {file_path}:{start_line}-{end_line}")
        if descriptor.code:
            lines.append(f"\nCode:\n{descriptor.code}")
        return "\n".join(lines)

    @mcp.tool()
    def find_type_definition(
        type_name: str,
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """Get a ClassDescriptor for a class, interface, or struct by name.

        Returns the type's signature, description, location, and all methods
        whose line range falls within the type's line range.

        Args:
            type_name: The class/interface/struct name to look up.
            workspace_id: Workspace to search (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        conn = sqlite3.connect(storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT symbol_name, symbol_type, file_path, start_line, end_line
            FROM symbol_index
            WHERE workspace_id = ? AND symbol_name = ?
              AND symbol_type IN ('class', 'interface', 'struct', 'type')
            LIMIT 1
            """,
            (workspace_id, type_name),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(
                f"Type '{type_name}' not found in workspace '{workspace_id}'"
            )

        file_path = row["file_path"]
        start_line = row["start_line"]
        end_line = row["end_line"]
        symbol_type = row["symbol_type"]

        cursor.execute(
            """
            SELECT code, docstring FROM code_fts
            WHERE workspace_id = ? AND file_path = ? AND symbol_name = ?
            LIMIT 1
            """,
            (workspace_id, file_path, type_name),
        )
        fts_row = cursor.fetchone()

        # Find methods whose line range is enclosed within this type's range
        cursor.execute(
            """
            SELECT symbol_name, symbol_type, start_line, end_line
            FROM symbol_index
            WHERE workspace_id = ? AND file_path = ? AND symbol_type = 'method'
              AND start_line >= ? AND end_line <= ?
            ORDER BY start_line
            """,
            (workspace_id, file_path, start_line, end_line),
        )
        method_rows = cursor.fetchall()
        conn.close()

        docstring = fts_row["docstring"] if fts_row else ""
        meta = _get_chunk_meta(storage, workspace_id, file_path, start_line, end_line)
        language = meta.get("language") or detect_language(file_path) or "unknown"
        signature = meta.get("signature", "")
        namespace = meta.get("package") or None

        methods = [
            FunctionDescriptor(
                language=language,
                kind="method",
                name=m["symbol_name"],
                location=SymbolLocation(
                    file_path=file_path,
                    start_line=m["start_line"],
                    end_line=m["end_line"],
                ),
            )
            for m in method_rows
        ]

        descriptor = ClassDescriptor(
            language=language,
            kind=symbol_type,
            name=type_name,
            namespace=namespace,
            full_name=f"{namespace}.{type_name}" if namespace else type_name,
            signature=signature,
            description=docstring or None,
            location=SymbolLocation(
                file_path=file_path, start_line=start_line, end_line=end_line
            ),
            methods=methods,
        )

        if output_format == "json":
            return json.dumps(asdict(descriptor), indent=2)

        lines = [
            f"Type:     {descriptor.name}",
            f"Kind:     {descriptor.kind}",
            f"Language: {descriptor.language}",
        ]
        if descriptor.namespace:
            lines.append(f"Namespace: {descriptor.namespace}")
        if descriptor.signature:
            lines.append(f"Signature: {descriptor.signature}")
        if descriptor.description:
            lines.append(f"Description: {descriptor.description}")
        lines.append(f"Location: {file_path}:{start_line}-{end_line}")
        if methods:
            lines.append(f"\nMethods ({len(methods)}):")
            for m in methods:
                loc = m.location
                lines.append(f"  - {m.name}  ({loc.file_path}:{loc.start_line}-{loc.end_line})")
        return "\n".join(lines)

    @mcp.tool()
    def find_implementations(
        symbol_name: str,
        direction: str = "callers",
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """Find what calls a symbol (callers) or what a symbol calls (targets).

        Args:
            symbol_name: The symbol to look up.
            direction: "callers" — who calls this symbol;
                       "targets" — what this symbol calls/imports.
            workspace_id: Workspace to search (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        if direction not in ("callers", "targets"):
            raise ValueError(f"Invalid direction '{direction}'. Use 'callers' or 'targets'.")

        analyzer = DependencyAnalyzer(storage)
        if direction == "callers":
            results = analyzer.get_callers(symbol_name, workspace_id)
            sym_key = "source_symbol"
        else:
            results = analyzer.get_targets(symbol_name, workspace_id)
            sym_key = "target_symbol"

        if output_format == "json":
            return json.dumps(
                {"symbol": symbol_name, "direction": direction, "results": results},
                indent=2,
            )

        if not results:
            return f"No {direction} found for '{symbol_name}' in workspace '{workspace_id}'."

        lines = [f"{len(results)} {direction} of '{symbol_name}':"]
        for r in results:
            sym = r.get(sym_key, "")
            fp = r.get("file_path") or ""
            entry = f"  - {sym}"
            if fp:
                entry += f"  ({fp})"
            lines.append(entry)
        return "\n".join(lines)

    @mcp.tool()
    def get_code_context(
        file_path: str,
        line: int,
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """Find which function or class encloses a given line in a file.

        Searches indexed chunks for the file, finds the innermost symbol
        whose line range contains the requested line, and reports the context.

        Args:
            file_path: Absolute path of the source file.
            line: Line number to look up (1-indexed).
            workspace_id: Workspace to search (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        conn = sqlite3.connect(storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT symbol_name, symbol_type, start_line, end_line
            FROM symbol_index
            WHERE workspace_id = ? AND file_path = ?
              AND start_line <= ? AND end_line >= ?
            ORDER BY (end_line - start_line) ASC
            """,
            (workspace_id, file_path, line, line),
        )
        rows = cursor.fetchall()
        conn.close()

        if output_format == "json":
            return json.dumps(
                {
                    "file_path": file_path,
                    "line": line,
                    "enclosing_symbols": [dict(r) for r in rows],
                },
                indent=2,
            )

        if not rows:
            return (
                f"Line {line} in '{file_path}' is not inside any indexed symbol."
            )

        innermost = rows[0]
        lines = [
            f"Line {line} is inside {innermost['symbol_type']} `{innermost['symbol_name']}`",
            f"  ({file_path}:{innermost['start_line']}-{innermost['end_line']})",
        ]
        if len(rows) > 1:
            lines.append("\nEnclosing context:")
            for r in rows[1:]:
                lines.append(
                    f"  - {r['symbol_type']} `{r['symbol_name']}`"
                    f" ({r['start_line']}-{r['end_line']})"
                )
        return "\n".join(lines)

    @mcp.tool()
    def project_map(
        workspace_id: str = "default",
        map_format: str = "summary",
        output_format: str = "human",
    ) -> str:
        """Generate a project map showing files, languages, and structure.

        Args:
            workspace_id: Workspace to map (default "default").
            map_format: "summary" (default), "detailed", or "tree".
            output_format: Output format — "human" (default) or "json".
        """
        generator = MapGenerator(storage)
        result = generator.generate_project_map(workspace_id, format=map_format)

        if output_format == "json":
            return json.dumps(result, indent=2)

        summary = result.get("summary", {})
        lines = [
            f"Project Map  [{workspace_id}]",
            f"Files:  {summary.get('total_files', 0)}",
            f"Lines:  {summary.get('total_lines', 0)}",
        ]

        lang_stats = summary.get("stats_by_language", {})
        if lang_stats:
            lines.append("\nLanguages:")
            for lang, stats in lang_stats.items():
                lines.append(f"  {lang}: {stats.get('files', 0)} files")

        top_folders = summary.get("top_level_folders", [])
        if top_folders:
            lines.append(f"\nTop-level folders: {', '.join(top_folders)}")

        most_active = summary.get("most_active_files", [])
        if most_active:
            lines.append("\nMost active files:")
            for f in most_active[:5]:
                lines.append(f"  {f['path']}  ({f['symbol_count']} symbols)")

        last_indexed = summary.get("last_indexed")
        if last_indexed:
            lines.append(f"\nLast indexed: {last_indexed}")

        if "tree" in result:
            lines.append("\nFile Tree:")
            lines.append(result["tree"])
        elif "detailed" in result:
            lines.append("\nFiles:")
            for f in result["detailed"].get("files", [])[:20]:
                lines.append(
                    f"  {f['path']}  [{f['language']}]"
                    f"  ({f.get('functions', 0)} fn,"
                    f" {f.get('classes', 0)} cls,"
                    f" {f.get('methods', 0)} mth)"
                )

        return "\n".join(lines)

    @mcp.tool()
    def file_tree(
        workspace_id: str = "default",
        path_filter: Optional[str] = None,
        max_depth: int = 10,
        output_format: str = "human",
    ) -> str:
        """Generate an annotated file tree for a workspace.

        Args:
            workspace_id: Workspace to display (default "default").
            path_filter: Optional glob pattern to filter files (e.g. "*.py").
            max_depth: Maximum directory depth (default 10).
            output_format: Output format — "human" (default) or "json".
        """
        generator = MapGenerator(storage)
        tree = generator.generate_file_tree(
            workspace_id, path_filter=path_filter, max_depth=max_depth
        )

        if output_format == "json":
            return json.dumps({"workspace_id": workspace_id, "tree": tree}, indent=2)

        if not tree:
            return f"No files found in workspace '{workspace_id}'."
        return tree

    @mcp.tool()
    def list_package_exports(
        package_path: str,
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """List public (non-underscore-prefixed) symbols exported by a package.

        Args:
            package_path: Package path to query, e.g. "mcp_code_rag".
            workspace_id: Workspace to search (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        generator = MapGenerator(storage)
        exports = generator.list_package_exports(workspace_id, package_path)

        if output_format == "json":
            return json.dumps(
                {"package": package_path, "exports": sorted(exports)}, indent=2
            )

        if not exports:
            return f"No public exports found in package '{package_path}'."

        lines = [f"Public exports from '{package_path}' ({len(exports)} symbols):"]
        for sym in sorted(exports):
            lines.append(f"  - {sym}")
        return "\n".join(lines)

    @mcp.tool()
    def dependency_graph(
        workspace_id: str = "default",
        entry_point: Optional[str] = None,
        max_depth: Optional[int] = None,
        output_format: str = "human",
    ) -> str:
        """Build and display the dependency graph for a workspace.

        Args:
            workspace_id: Workspace to analyse (default "default").
            entry_point: Optional symbol name to root a BFS subgraph on.
            max_depth: Maximum BFS depth when entry_point is given.
            output_format: Output format — "human" (default) or "json".
        """
        analyzer = DependencyAnalyzer(storage)
        graph = analyzer.build_dependency_graph(
            workspace_id, max_depth=max_depth, entry_point=entry_point
        )

        if output_format == "json":
            serializable: dict = {
                "edges": graph["edges"],
                "adjacency": graph["adjacency"],
                "reverse_adjacency": graph["reverse_adjacency"],
                "visited_nodes": list(graph["visited_nodes"]),
            }
            if "entry_point_graph" in graph:
                ep = graph["entry_point_graph"]
                serializable["entry_point_graph"] = {
                    "nodes": list(ep["nodes"]),
                    "edges": ep["edges"],
                    "depths": ep["depths"],
                }
            return json.dumps(serializable, indent=2)

        summary = analyzer.format_summary(graph, workspace_id)
        lines = [
            f"Dependency Graph  [{workspace_id}]",
            f"Symbols:      {summary['total_symbols']}",
            f"Dependencies: {summary['total_dependencies']}",
        ]
        if summary["most_imported"]:
            lines.append(
                f"Most imported: {summary['most_imported']}"
                f"  ({summary['most_imported_count']} callers)"
            )
        if summary["most_dependent"]:
            lines.append(
                f"Most dependent: {summary['most_dependent']}"
                f"  ({summary['most_dependent_count']} deps)"
            )

        if entry_point and "entry_point_graph" in graph:
            ep = graph["entry_point_graph"]
            lines.append(f"\nSubgraph from '{entry_point}' ({len(ep['nodes'])} nodes):")
            for src, tgt in ep["edges"][:20]:
                lines.append(f"  {src}  ->  {tgt}")
        else:
            lines.append("")
            lines.append(analyzer.format_visual(graph))

        return "\n".join(lines)

    # ── Phase 3 tools ────────────────────────────────────────────────────────

    @mcp.tool()
    def watch_directory(
        directory: str,
        workspace_id: str = "default",
        recursive: bool = True,
        enabled: bool = True,
        format: str = "human",
    ) -> str:
        """Watch a directory for file changes and automatically re-index.

        When enabled=True, starts monitoring the directory with FileWatcher.
        When enabled=False, stops an active watcher (directory ignored if watcher_id provided).

        Args:
            directory: Absolute or relative path to the directory to watch.
            workspace_id: Workspace identifier (default "default").
            recursive: If True, watch subdirectories (default True).
            enabled: If True, start watching; if False, stop watching (default True).
            format: Output format — "human" (default) or "json".

        Returns:
            Status message with watcher_id if enabled, or confirmation if stopped.
        """
        if enabled:
            # Validate directory
            dir_path = Path(directory)
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {directory}")
            if not dir_path.is_dir():
                raise ValueError(f"Not a directory: {directory}")

            # Start watching
            watcher_id = file_watcher.start(
                str(dir_path), workspace_id, recursive=recursive
            )

            result = {
                "status": "watching",
                "watcher_id": watcher_id,
                "directory": str(dir_path),
                "workspace_id": workspace_id,
                "recursive": recursive,
            }

            if format == "json":
                return json.dumps(result, indent=2)

            return f"Started watching {dir_path} (ID: {watcher_id})"
        else:
            # List all active watchers and stop them for this directory (or all if directory not found)
            active_watchers = file_watcher.list_watchers()
            stopped_count = 0

            for watcher in active_watchers:
                if watcher["workspace_id"] == workspace_id:
                    if file_watcher.stop(watcher["id"]):
                        stopped_count += 1

            result = {
                "status": "stopped",
                "watchers_stopped": stopped_count,
                "workspace_id": workspace_id,
            }

            if format == "json":
                return json.dumps(result, indent=2)

            return f"Stopped {stopped_count} watcher(s) for workspace '{workspace_id}'"

    @mcp.tool()
    def reindex_file(
        file_path: str,
        workspace_id: str = "default",
        format: str = "human",
    ) -> str:
        """Re-index a single file with force_reindex=True.

        Forces the file to be re-chunked, re-embedded, and re-stored regardless
        of hash cache. Applies both H1 and H2 tagging.

        Args:
            file_path: Absolute or relative path to the file.
            workspace_id: Workspace identifier (default "default").
            format: Output format — "human" (default) or "json".

        Returns:
            Statistics about the re-indexing operation.
        """
        file_path_abs = str(Path(file_path).resolve())
        if not Path(file_path_abs).exists():
            raise ValueError(f"File not found: {file_path}")
        if not Path(file_path_abs).is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Ingest with force_reindex=True
        result = pipeline.ingest_file(file_path_abs, workspace_id, force_reindex=True)

        output = {
            "file_path": file_path_abs,
            "workspace_id": workspace_id,
            "indexed": result["indexed"],
            "chunks_created": result["chunks_created"],
            "skipped": result["skipped"],
            "errors": result["errors"],
        }

        if format == "json":
            return json.dumps(output, indent=2)

        lines = [f"Re-indexed {file_path_abs}"]
        if result["indexed"]:
            lines.append(f"  Chunks created: {result['chunks_created']}")
        elif result["skipped"]:
            lines.append("  File was skipped")
        if result["errors"]:
            lines.append("  Errors:")
            for err in result["errors"]:
                lines.append(f"    - {err}")
        return "\n".join(lines)

    @mcp.tool()
    def project_stats(
        workspace_id: str = "default",
        output_format: str = "human",
    ) -> str:
        """Get project statistics with scan history and evolution delta.

        Args:
            workspace_id: Workspace to report on (default "default").
            output_format: Output format — "human" (default) or "json".
        """
        generator = MapGenerator(storage)
        stats = generator.generate_project_stats(workspace_id)

        if output_format == "json":
            return json.dumps(stats, indent=2)

        current = stats.get("current", {})
        delta = stats.get("delta", {})
        scans = stats.get("scan_history", [])

        lines = [
            f"Project Stats  [{workspace_id}]",
            f"Files:         {current.get('files', 0)}",
            f"Total lines:   {current.get('total_lines', 0)}",
            f"Total symbols: {current.get('total_symbols', 0)}",
            f"Scans:         {len(scans)}",
        ]

        if delta:
            files_delta = delta.get("files_delta", 0)
            chunks_delta = delta.get("chunks_delta", 0)
            lines.append("\nEvolution (first → last scan):")
            lines.append(f"  Files delta:  {'+' if files_delta >= 0 else ''}{files_delta}")
            lines.append(
                f"  Chunks delta: {'+' if chunks_delta >= 0 else ''}{chunks_delta}"
            )
            time_span = delta.get("time_span_ms", 0)
            if time_span:
                lines.append(f"  Time span:    {time_span / 1000:.1f}s")

        if scans:
            lines.append(f"\nLatest scan: {scans[-1].get('timestamp', 'N/A')}")

        return "\n".join(lines)

    return mcp


def init_services(
    config: AppConfig,
) -> tuple[Storage, OllamaClient, CodeIngestPipeline, HybridSearch]:
    """Initialise all service singletons from config."""
    storage = Storage(config.rag.index_path)
    ollama_client = OllamaClient(config.ollama)
    pipeline = CodeIngestPipeline(storage, ollama_client, config)
    search = HybridSearch(storage, ollama_client, config.hybrid_search)
    return storage, ollama_client, pipeline, search


def main() -> None:
    """CLI entry point (wired in pyproject.toml)."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Code RAG Server")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--transport",
        default="sse",
        choices=["sse", "stdio", "http", "streamable-http"],
    )
    parser.add_argument("--port", type=int, default=3005)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.logging.level,
        format=config.logging.format,
        file=config.logging.file or None,
    )

    storage, ollama_client, pipeline, search = init_services(config)
    server = create_server(config, storage, ollama_client, pipeline, search)

    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
