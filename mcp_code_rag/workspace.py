"""Workspace auto-detection and management for MCP Code RAG."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mcp_code_rag.utils.hashing import short_hash


WORKSPACE_MARKERS = [
    "pyproject.toml",
    "setup.py",
    "go.mod",
    "package.json",
    "Cargo.toml",
    "pom.xml",
    ".csproj",
    "Gemfile",
    "composer.json",
]

MARKER_LANGUAGES = {
    "pyproject.toml": ["python"],
    "setup.py": ["python"],
    "go.mod": ["go"],
    "package.json": ["javascript", "typescript"],
    "Cargo.toml": ["rust"],
    "pom.xml": ["java"],
    ".csproj": ["csharp"],
    "Gemfile": ["ruby"],
    "composer.json": ["php"],
}


@dataclass
class WorkspaceInfo:
    """Information about a detected workspace."""

    id: str
    name: str
    root: str
    marker_type: str
    languages: list[str]


def detect_workspace(dir_path: str) -> Optional[WorkspaceInfo]:
    """
    Detect workspace by scanning up directory tree for markers.

    Scans from dir_path upwards looking for workspace markers in order:
    pyproject.toml > setup.py > go.mod > package.json > Cargo.toml >
    pom.xml > .csproj > Gemfile > composer.json

    Args:
        dir_path: Starting directory path (can be absolute or relative).

    Returns:
        WorkspaceInfo if marker found, None otherwise.
    """
    current = Path(dir_path).resolve()

    while current != current.parent:
        for marker in WORKSPACE_MARKERS:
            marker_path = current / marker
            if marker_path.exists():
                workspace_id = get_workspace_id(str(current))
                languages = MARKER_LANGUAGES.get(marker, [])
                return WorkspaceInfo(
                    id=workspace_id,
                    name=current.name,
                    root=str(current),
                    marker_type=marker,
                    languages=languages,
                )
        current = current.parent

    return None


def get_workspace_id(root_path: str) -> str:
    """
    Generate stable 12-character workspace ID from root path.

    Args:
        root_path: Absolute path to workspace root.

    Returns:
        12-character hexadecimal workspace ID.
    """
    abs_path = Path(root_path).resolve()
    return short_hash(str(abs_path), length=12)


def get_collection_name(workspace_id: str, lang: str) -> str:
    """
    Format collection name for semantic search index.

    Args:
        workspace_id: Workspace ID (typically 12 characters).
        lang: Programming language identifier.

    Returns:
        Collection name in format: coderag-{workspace_id}-{lang}
    """
    return f"coderag-{workspace_id}-{lang}"


def _init_cache_table(conn: sqlite3.Connection) -> None:
    """Initialize workspace cache table if not exists."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspace_cache (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            root TEXT NOT NULL UNIQUE,
            marker_type TEXT NOT NULL,
            languages TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def cache_workspace(db_path: str, workspace_info: WorkspaceInfo) -> None:
    """
    Cache workspace information in SQLite database.

    Args:
        db_path: Path to SQLite database.
        workspace_info: WorkspaceInfo to cache.
    """
    conn = sqlite3.connect(db_path)
    try:
        _init_cache_table(conn)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO workspace_cache
            (id, name, root, marker_type, languages)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                workspace_info.id,
                workspace_info.name,
                workspace_info.root,
                workspace_info.marker_type,
                ",".join(workspace_info.languages),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_workspace(db_path: str, workspace_id: str) -> Optional[WorkspaceInfo]:
    """
    Retrieve cached workspace information from SQLite database.

    Args:
        db_path: Path to SQLite database.
        workspace_id: Workspace ID to look up.

    Returns:
        WorkspaceInfo if found in cache, None otherwise.
    """
    conn = sqlite3.connect(db_path)
    try:
        _init_cache_table(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, root, marker_type, languages FROM workspace_cache WHERE id = ?",
            (workspace_id,),
        )
        row = cursor.fetchone()

        if row:
            return WorkspaceInfo(
                id=row[0],
                name=row[1],
                root=row[2],
                marker_type=row[3],
                languages=row[4].split(","),
            )
        return None
    finally:
        conn.close()


def list_workspaces(db_path: str) -> list[WorkspaceInfo]:
    """
    List all cached workspaces from SQLite database.

    Args:
        db_path: Path to SQLite database.

    Returns:
        List of cached WorkspaceInfo objects.
    """
    conn = sqlite3.connect(db_path)
    try:
        _init_cache_table(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, root, marker_type, languages FROM workspace_cache ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()

        workspaces = []
        for row in rows:
            workspaces.append(
                WorkspaceInfo(
                    id=row[0],
                    name=row[1],
                    root=row[2],
                    marker_type=row[3],
                    languages=row[4].split(","),
                )
            )
        return workspaces
    finally:
        conn.close()
