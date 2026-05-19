"""Storage layer encapsulating ChromaDB + SQLite for code indexing."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Any

import chromadb

from mcp_code_rag.chunker import CodeChunk


class Storage:
    """Encapsulates ChromaDB vector store and SQLite relational database."""

    def __init__(self, index_path: str):
        """
        Initialize Storage with ChromaDB and SQLite.

        Creates index directory if missing, initializes ChromaDB client,
        and creates SQLite tables (path_index, symbol_index, dependency_index,
        scan_history, hash_cache, workspace_cache, code_fts).

        Args:
            index_path: Directory path for ChromaDB and SQLite storage.
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB (new API)
        chroma_path = self.index_path / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))

        # Initialize SQLite
        db_path = self.index_path / "storage.db"
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create SQLite tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # path_index: maps file_path -> workspace_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS path_index (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                workspace_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # symbol_index: maps symbol names -> chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_index (
                id INTEGER PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,  -- "function", "class", "method", etc.
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                tags TEXT,  -- JSON list of tags
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, file_path, start_line, end_line)
            )
        """)

        # dependency_index: tracks dependencies between symbols
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependency_index (
                id INTEGER PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_symbol TEXT NOT NULL,
                target_symbol TEXT NOT NULL,
                symbols TEXT,  -- JSON list of symbols
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, source_symbol, target_symbol)
            )
        """)

        # scan_history: tracks scan events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                files_scanned INTEGER,
                chunks_created INTEGER,
                duration_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # hash_cache: caches file hashes for change detection
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hash_cache (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # workspace_cache: caches workspace metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_cache (
                id INTEGER PRIMARY KEY,
                workspace_id TEXT NOT NULL UNIQUE,
                metadata TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # code_fts: Full-Text Search table for code
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(
                workspace_id,
                file_path,
                symbol_name,
                code,
                docstring
            )
        """)

        conn.commit()
        conn.close()

    def get_or_create_collection(
        self, workspace_id: str, lang: str
    ) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection for a workspace and language.

        Args:
            workspace_id: Unique identifier for the workspace.
            lang: Programming language ("python", "go", "javascript", etc.).

        Returns:
            ChromaDB Collection object.
        """
        collection_name = f"{workspace_id}_{lang}".replace(".", "_")
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"workspace_id": workspace_id, "language": lang},
        )

    def store_chunk(
        self, chunk: CodeChunk, embedding: list[float], workspace_id: str
    ) -> None:
        """
        Store a chunk in ChromaDB and SQLite indexes.

        Inserts into symbol_index and code_fts. Updates path_index.

        Args:
            chunk: CodeChunk object to store.
            embedding: Vector embedding (typically 768-1536 dims).
            workspace_id: Workspace identifier.
        """
        # Store in ChromaDB
        collection = self.get_or_create_collection(workspace_id, chunk.language)
        chunk_id = f"{workspace_id}_{chunk.file_path}_{chunk.start_line}_{chunk.end_line}"

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk.to_embed_text()],
            metadatas=[chunk.to_chromadb_metadata()],
        )

        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Update path_index
        cursor.execute(
            """
            INSERT OR IGNORE INTO path_index (file_path, workspace_id)
            VALUES (?, ?)
            """,
            (chunk.file_path, workspace_id),
        )

        # Insert into symbol_index
        tags_json = json.dumps(chunk.metadata.get("tags", [])) if "tags" in chunk.metadata else None
        cursor.execute(
            """
            INSERT OR REPLACE INTO symbol_index
            (workspace_id, symbol_name, symbol_type, file_path, start_line, end_line, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                chunk.name,
                chunk.type,
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
                tags_json,
            ),
        )

        # Insert into code_fts
        cursor.execute(
            """
            INSERT INTO code_fts
            (workspace_id, file_path, symbol_name, code, docstring)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                chunk.file_path,
                chunk.name,
                chunk.code,
                chunk.docstring,
            ),
        )

        conn.commit()
        conn.close()

    def delete_file_chunks(self, file_path: str, workspace_id: str) -> None:
        """
        Delete all chunks for a file from ChromaDB and SQLite.

        Args:
            file_path: Absolute path to the file.
            workspace_id: Workspace identifier.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all chunks for this file to find collection names
        cursor.execute(
            "SELECT DISTINCT symbol_type FROM symbol_index WHERE file_path = ? AND workspace_id = ?",
            (file_path, workspace_id),
        )
        # Since we need language info, query chromadb collections
        # Get symbols first
        cursor.execute(
            "SELECT * FROM symbol_index WHERE file_path = ? AND workspace_id = ?",
            (file_path, workspace_id),
        )
        symbols = cursor.fetchall()

        # Delete from ChromaDB (try all collections)
        try:
            collections = self.chroma_client.list_collections()
            for coll in collections:
                if workspace_id in coll.name:
                    collection = self.chroma_client.get_collection(coll.name)
                    # Get all IDs for this file
                    results = collection.get(
                        where={"file_path": file_path}, include=[]
                    )
                    if results["ids"]:
                        collection.delete(ids=results["ids"])
        except Exception:
            pass

        # Delete from SQLite indexes
        cursor.execute(
            "DELETE FROM symbol_index WHERE file_path = ? AND workspace_id = ?",
            (file_path, workspace_id),
        )
        cursor.execute(
            "DELETE FROM code_fts WHERE file_path = ? AND workspace_id = ?",
            (file_path, workspace_id),
        )

        conn.commit()
        conn.close()

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """
        Retrieve cached file hash.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Cached hash string if exists, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_hash FROM hash_cache WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def set_file_hash(self, file_path: str, file_hash: str) -> None:
        """
        Store file hash in cache.

        Args:
            file_path: Absolute path to the file.
            file_hash: Hash string (e.g., SHA256).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO hash_cache (file_path, file_hash) VALUES (?, ?)",
            (file_path, file_hash),
        )
        conn.commit()
        conn.close()

    def store_dependency(
        self,
        source: str,
        target: str,
        workspace_id: str,
        symbols: Optional[list[str]] = None,
    ) -> None:
        """
        Record a dependency between two symbols.

        Args:
            source: Source symbol identifier.
            target: Target symbol identifier.
            workspace_id: Workspace identifier.
            symbols: Optional list of intermediate symbols.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        symbols_json = json.dumps(symbols) if symbols else None
        cursor.execute(
            """
            INSERT OR REPLACE INTO dependency_index
            (workspace_id, source_symbol, target_symbol, symbols)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, source, target, symbols_json),
        )
        conn.commit()
        conn.close()

    def get_dependencies(
        self, symbol: str, workspace_id: str
    ) -> list[dict[str, Any]]:
        """
        Query dependencies for a symbol.

        Args:
            symbol: Symbol identifier.
            workspace_id: Workspace identifier.

        Returns:
            List of dependency records.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM dependency_index
            WHERE (source_symbol = ? OR target_symbol = ?) AND workspace_id = ?
            """,
            (symbol, symbol, workspace_id),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def record_scan(
        self, workspace_id: str, files_scanned: int, chunks_created: int, duration_ms: int
    ) -> None:
        """
        Record a scan event in scan history.

        Args:
            workspace_id: Workspace identifier.
            files_scanned: Number of files processed.
            chunks_created: Number of chunks created.
            duration_ms: Scan duration in milliseconds.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scan_history
            (workspace_id, files_scanned, chunks_created, duration_ms)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, files_scanned, chunks_created, duration_ms),
        )
        conn.commit()
        conn.close()

    def delete_workspace(self, workspace_id: str) -> None:
        """
        Delete all data for a workspace from ChromaDB and SQLite.

        Removes ChromaDB collections and all related SQLite entries.

        Args:
            workspace_id: Workspace identifier.
        """
        # Delete from ChromaDB
        try:
            collections = self.chroma_client.list_collections()
            for coll in collections:
                if workspace_id in coll.name:
                    self.chroma_client.delete_collection(coll.name)
        except Exception:
            pass

        # Delete from SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM path_index WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM symbol_index WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM dependency_index WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM scan_history WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM workspace_cache WHERE workspace_id = ?", (workspace_id,))
        cursor.execute("DELETE FROM code_fts WHERE workspace_id = ?", (workspace_id,))

        conn.commit()
        conn.close()

    def close(self) -> None:
        """Close storage connections."""
        # ChromaDB client is stateless; just clear reference
        self.chroma_client = None
