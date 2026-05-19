"""Generate project maps and statistics from indexed code."""

import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp_code_rag.storage import Storage


class MapGenerator:
    """Generate project maps and statistics from indexed code."""

    def __init__(self, storage: Storage):
        """
        Initialize MapGenerator with a Storage instance.

        Args:
            storage: Storage instance for accessing indexed data.
        """
        self.storage = storage

    def generate_project_map(
        self, workspace_id: str, format: str = "summary"
    ) -> dict[str, Any]:
        """
        Generate a project map in the specified format.

        Args:
            workspace_id: Workspace identifier.
            format: Map format - "summary", "detailed", or "tree".

        Returns:
            Dictionary with project information based on format:
            - "summary": total_files, total_lines, stats by language, top_level_folders, most_active_files, last_indexed
            - "detailed": all of summary + file listing with stats
            - "tree": text-based annotated file tree with LOC/functions
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all files for this workspace
        cursor.execute(
            "SELECT file_path FROM path_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        files = [row["file_path"] for row in cursor.fetchall()]

        # Get symbol statistics
        cursor.execute(
            """
            SELECT file_path, symbol_type, COUNT(*) as count
            FROM symbol_index
            WHERE workspace_id = ?
            GROUP BY file_path, symbol_type
            """,
            (workspace_id,),
        )
        file_stats = defaultdict(lambda: {"functions": 0, "classes": 0, "methods": 0})
        # Map singular DB types to plural dict keys
        type_map = {"function": "functions", "class": "classes", "method": "methods"}
        for row in cursor.fetchall():
            file_path = row["file_path"]
            symbol_type = row["symbol_type"]
            count = row["count"]
            plural = type_map.get(symbol_type, symbol_type)
            if plural in file_stats[file_path]:
                file_stats[file_path][plural] = count

        # Calculate language stats
        lang_stats = defaultdict(
            lambda: {"files": 0, "functions": 0, "classes": 0, "methods": 0}
        )
        for file_path in files:
            lang = self._infer_language(file_path)
            lang_stats[lang]["files"] += 1

        for file_path, stats in file_stats.items():
            lang = self._infer_language(file_path)
            for symbol_type in ("functions", "classes", "methods"):
                lang_stats[lang][symbol_type] += stats.get(symbol_type, 0)

        # Get top-level folders
        top_level_folders = self._get_top_level_folders(files)

        # Get most active files
        cursor.execute(
            """
            SELECT file_path, COUNT(*) as symbol_count
            FROM symbol_index
            WHERE workspace_id = ?
            GROUP BY file_path
            ORDER BY symbol_count DESC
            LIMIT 10
            """,
            (workspace_id,),
        )
        most_active_files = [
            {"path": row["file_path"], "symbol_count": row["symbol_count"]}
            for row in cursor.fetchall()
        ]

        # Get last indexed timestamp
        cursor.execute(
            """
            SELECT timestamp
            FROM scan_history
            WHERE workspace_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (workspace_id,),
        )
        last_scan = cursor.fetchone()
        last_indexed = last_scan["timestamp"] if last_scan else None

        # Calculate total lines (approximate from code length)
        cursor.execute(
            """
            SELECT SUM(end_line - start_line + 1) as total_lines
            FROM symbol_index
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        total_lines = cursor.fetchone()["total_lines"] or 0

        conn.close()

        result = {
            "summary": {
                "total_files": len(files),
                "total_lines": total_lines,
                "stats_by_language": dict(lang_stats),
                "top_level_folders": top_level_folders,
                "most_active_files": most_active_files,
                "last_indexed": last_indexed,
            }
        }

        if format == "summary":
            return result

        if format == "detailed":
            file_list = []
            for file_path in files:
                stats = file_stats.get(file_path, {})
                file_list.append(
                    {
                        "path": file_path,
                        "language": self._infer_language(file_path),
                        "functions": stats.get("functions", 0),
                        "classes": stats.get("classes", 0),
                        "methods": stats.get("methods", 0),
                    }
                )
            result["detailed"] = {"files": file_list}
            return result

        if format == "tree":
            tree = self.generate_file_tree(workspace_id)
            result["tree"] = tree
            return result

        return result

    def generate_file_tree(
        self, workspace_id: str, path_filter: Optional[str] = None, max_depth: int = 10
    ) -> str:
        """
        Generate a filtered file tree as text with LOC and function annotations.

        Args:
            workspace_id: Workspace identifier.
            path_filter: Optional glob pattern to filter files.
            max_depth: Maximum directory depth to display.

        Returns:
            Annotated file tree as text.
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_path FROM path_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        files = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Filter files if needed
        if path_filter:
            from fnmatch import fnmatch

            files = [f for f in files if fnmatch(f, path_filter)]

        # Build tree structure
        tree_dict = {}
        for file_path in files:
            parts = Path(file_path).parts
            current = tree_dict
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            if len(parts) > 0:
                current[parts[-1]] = {"_file": True, "_path": file_path}

        # Get symbol counts for files
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT file_path, COUNT(*) as count
            FROM symbol_index
            WHERE workspace_id = ?
            GROUP BY file_path
            """,
            (workspace_id,),
        )
        symbol_counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        # Render tree
        lines = []
        self._render_tree(tree_dict, "", 0, max_depth, symbol_counts, lines)
        return "\n".join(lines)

    def _render_tree(
        self,
        tree: dict,
        prefix: str,
        depth: int,
        max_depth: int,
        symbol_counts: dict[str, int],
        lines: list[str],
    ) -> None:
        """Recursively render tree structure."""
        if depth > max_depth:
            return

        items = sorted(tree.items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if isinstance(value, dict):
                if value.get("_file"):
                    # It's a file — use stored absolute path for symbol lookup
                    abs_path = value.get("_path", key)
                    count = symbol_counts.get(abs_path, 0)
                    lines.append(f"{prefix}{connector}{key} ({count} symbols)")
                else:
                    # It's a directory
                    lines.append(f"{prefix}{connector}{key}/")
                    self._render_tree(
                        value, next_prefix, depth + 1, max_depth, symbol_counts, lines
                    )

    def generate_project_stats(self, workspace_id: str) -> dict[str, Any]:
        """
        Generate project statistics with historical evolution.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Dictionary with stats and delta between scans.
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all scans for this workspace
        cursor.execute(
            """
            SELECT files_scanned, chunks_created, duration_ms, timestamp
            FROM scan_history
            WHERE workspace_id = ?
            ORDER BY timestamp ASC
            """,
            (workspace_id,),
        )
        scans = [dict(row) for row in cursor.fetchall()]

        # Current stats
        cursor.execute(
            """
            SELECT COUNT(DISTINCT file_path) as file_count,
                   SUM(end_line - start_line + 1) as total_lines,
                   COUNT(*) as total_symbols
            FROM symbol_index
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        current = cursor.fetchone()
        current_stats = {
            "files": current["file_count"] or 0,
            "total_lines": current["total_lines"] or 0,
            "total_symbols": current["total_symbols"] or 0,
        }

        # Language breakdown
        cursor.execute(
            """
            SELECT si.file_path, COUNT(*) as count
            FROM symbol_index si
            WHERE si.workspace_id = ?
            GROUP BY si.file_path
            """,
            (workspace_id,),
        )
        file_symbols = {}
        for row in cursor.fetchall():
            file_symbols[row[0]] = row[1]

        # Calculate delta
        delta = {}
        if len(scans) >= 2:
            first_scan = scans[0]
            last_scan = scans[-1]
            delta = {
                "files_delta": last_scan["files_scanned"] - first_scan["files_scanned"],
                "chunks_delta": last_scan["chunks_created"]
                - first_scan["chunks_created"],
                "time_span_ms": (
                    datetime.fromisoformat(last_scan["timestamp"])
                    - datetime.fromisoformat(first_scan["timestamp"])
                ).total_seconds()
                * 1000,
            }

        conn.close()

        return {
            "current": current_stats,
            "scan_history": scans,
            "delta": delta,
        }

    def list_package_exports(self, workspace_id: str, package_path: str) -> list[str]:
        """
        List public symbols (non-underscore-prefixed) from a package.

        Args:
            workspace_id: Workspace identifier.
            package_path: Package path to query (e.g., "mcp_code_rag").

        Returns:
            List of public symbol names.
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        # Get all symbols, then filter by package using file paths
        cursor.execute(
            """
            SELECT DISTINCT symbol_name, file_path
            FROM symbol_index
            WHERE workspace_id = ?
            """,
            (workspace_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        # Filter symbols by package path in file_path and exclude private symbols
        symbols = []
        for symbol_name, file_path in rows:
            if package_path in file_path and not symbol_name.startswith("_"):
                symbols.append(symbol_name)

        return list(set(symbols))

    def _infer_language(self, file_path: str) -> str:
        """Infer language from file extension."""
        from mcp_code_rag.extractors import detect_language

        lang = detect_language(file_path)
        return lang or "unknown"

    def _get_top_level_folders(self, files: list[str]) -> list[str]:
        """Extract top-level folders (first level under root) from file paths."""
        folders = set()
        for file_path in files:
            parts = Path(file_path).parts
            # For absolute paths, skip "/" (parts[0]) and get the first folder (parts[1])
            if len(parts) > 1 and parts[0] == "/":
                folders.add(parts[1])
            elif len(parts) > 0 and parts[0] != "/":
                # For relative paths, just add the first part
                folders.add(parts[0])
        return sorted(folders)
