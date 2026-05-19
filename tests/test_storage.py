"""Tests for storage layer (ChromaDB + SQLite)."""

import sqlite3
from pathlib import Path

import pytest

from mcp_code_rag.chunker import CodeChunk
from mcp_code_rag.storage import Storage


@pytest.fixture
def storage(tmp_path):
    """Create a Storage instance with temp directory."""
    return Storage(str(tmp_path))


@pytest.fixture
def sample_chunk():
    """Create a sample CodeChunk for testing."""
    return CodeChunk(
        type="function",
        name="hello_world",
        package="example",
        language="python",
        file_path="/workspace/example.py",
        start_line=10,
        end_line=15,
        selection_start=10,
        selection_end=10,
        signature="def hello_world(name: str) -> str:",
        docstring="Greet a person.",
        code='def hello_world(name: str) -> str:\n    return f"Hello {name}"',
        metadata={"visibility": "public"},
    )


class TestStoreChunk:
    """Tests for store_chunk method."""

    def test_store_chunk_in_chromadb(self, storage, sample_chunk):
        """Verify chunk is stored in ChromaDB."""
        workspace_id = "ws1"
        embedding = [0.1] * 768  # Mock 768-dim embedding

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Verify in ChromaDB
        collection = storage.get_or_create_collection(workspace_id, "python")
        results = collection.get()
        assert len(results["ids"]) > 0
        assert results["documents"][0] is not None

    def test_store_chunk_in_symbol_index(self, storage, sample_chunk):
        """Verify chunk is indexed in symbol_index."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Query symbol_index directly
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol_name, symbol_type FROM symbol_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "hello_world"
        assert row[1] == "function"

    def test_store_chunk_in_fts(self, storage, sample_chunk):
        """Verify chunk is indexed in full-text search."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Query code_fts
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM code_fts WHERE workspace_id = ?",
            (workspace_id,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None

    def test_store_chunk_updates_path_index(self, storage, sample_chunk):
        """Verify chunk storage updates path_index."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Query path_index
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT workspace_id FROM path_index WHERE file_path = ?",
            (sample_chunk.file_path,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == workspace_id


class TestHashCache:
    """Tests for file hash caching."""

    def test_set_and_get_file_hash(self, storage):
        """Verify hash cache round-trip."""
        file_path = "/workspace/example.py"
        file_hash = "abc123def456"

        storage.set_file_hash(file_path, file_hash)
        retrieved = storage.get_file_hash(file_path)

        assert retrieved == file_hash

    def test_get_nonexistent_hash_returns_none(self, storage):
        """Verify getting nonexistent hash returns None."""
        retrieved = storage.get_file_hash("/nonexistent/file.py")
        assert retrieved is None

    def test_hash_cache_update(self, storage):
        """Verify hash cache can be updated."""
        file_path = "/workspace/example.py"
        storage.set_file_hash(file_path, "hash1")
        storage.set_file_hash(file_path, "hash2")

        retrieved = storage.get_file_hash(file_path)
        assert retrieved == "hash2"


class TestDeleteFileChunks:
    """Tests for delete_file_chunks method."""

    def test_delete_file_chunks_from_symbol_index(self, storage, sample_chunk):
        """Verify delete_file_chunks removes from symbol_index."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Verify it's there
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM symbol_index WHERE file_path = ?",
            (sample_chunk.file_path,),
        )
        count_before = cursor.fetchone()[0]
        conn.close()
        assert count_before > 0

        # Delete
        storage.delete_file_chunks(sample_chunk.file_path, workspace_id)

        # Verify it's gone
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM symbol_index WHERE file_path = ?",
            (sample_chunk.file_path,),
        )
        count_after = cursor.fetchone()[0]
        conn.close()
        assert count_after == 0

    def test_delete_file_chunks_from_fts(self, storage, sample_chunk):
        """Verify delete_file_chunks removes from code_fts."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Delete
        storage.delete_file_chunks(sample_chunk.file_path, workspace_id)

        # Verify it's gone from FTS
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM code_fts WHERE file_path = ?",
            (sample_chunk.file_path,),
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0


class TestStoreDependency:
    """Tests for store_dependency method."""

    def test_store_dependency(self, storage):
        """Verify dependency is stored."""
        workspace_id = "ws1"
        source = "module_a.func1"
        target = "module_b.func2"
        symbols = ["helper1", "helper2"]

        storage.store_dependency(source, target, workspace_id, symbols)

        # Query directly
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM dependency_index WHERE source_symbol = ? AND target_symbol = ?",
            (source, target),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None

    def test_get_dependencies(self, storage):
        """Verify get_dependencies retrieves stored dependencies."""
        workspace_id = "ws1"
        source = "module_a.func1"
        target = "module_b.func2"

        storage.store_dependency(source, target, workspace_id)

        deps = storage.get_dependencies(source, workspace_id)
        assert len(deps) > 0
        assert deps[0]["source_symbol"] == source


class TestDeleteWorkspace:
    """Tests for delete_workspace method."""

    def test_delete_workspace_clears_symbol_index(self, storage, sample_chunk):
        """Verify delete_workspace removes all symbol_index entries."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        # Verify data exists
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM symbol_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        count_before = cursor.fetchone()[0]
        conn.close()
        assert count_before > 0

        # Delete workspace
        storage.delete_workspace(workspace_id)

        # Verify all gone
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM symbol_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        count_after = cursor.fetchone()[0]
        conn.close()
        assert count_after == 0

    def test_delete_workspace_clears_path_index(self, storage, sample_chunk):
        """Verify delete_workspace removes path_index entries."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, workspace_id)

        storage.delete_workspace(workspace_id)

        # Verify path_index is cleared
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM path_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

    def test_delete_workspace_clears_dependencies(self, storage):
        """Verify delete_workspace removes all dependencies."""
        workspace_id = "ws1"
        storage.store_dependency("a", "b", workspace_id)
        storage.store_dependency("c", "d", workspace_id)

        storage.delete_workspace(workspace_id)

        # Verify all dependencies are gone
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM dependency_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0


class TestMultipleWorkspaces:
    """Tests for isolation between workspaces."""

    def test_chunks_isolated_by_workspace(self, storage, sample_chunk):
        """Verify chunks in one workspace don't leak to another."""
        ws1 = "ws1"
        ws2 = "ws2"
        embedding = [0.1] * 768

        storage.store_chunk(sample_chunk, embedding, ws1)

        # Query ws1 collection
        coll1 = storage.get_or_create_collection(ws1, "python")
        results1 = coll1.get()
        assert len(results1["ids"]) > 0

        # Query ws2 collection (should be empty)
        coll2 = storage.get_or_create_collection(ws2, "python")
        results2 = coll2.get()
        assert len(results2["ids"]) == 0

    def test_dependencies_isolated_by_workspace(self, storage):
        """Verify dependencies are isolated by workspace."""
        ws1 = "ws1"
        ws2 = "ws2"

        storage.store_dependency("a", "b", ws1)
        storage.store_dependency("c", "d", ws2)

        deps_ws1 = storage.get_dependencies("a", ws1)
        deps_ws2 = storage.get_dependencies("a", ws2)

        assert len(deps_ws1) > 0
        assert len(deps_ws2) == 0


class TestRecordScan:
    """Tests for scan history recording."""

    def test_record_scan(self, storage):
        """Verify scan history is recorded."""
        workspace_id = "ws1"
        files_scanned = 42
        chunks_created = 128
        duration_ms = 5000

        storage.record_scan(workspace_id, files_scanned, chunks_created, duration_ms)

        # Query scan_history
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT files_scanned, chunks_created, duration_ms FROM scan_history WHERE workspace_id = ?",
            (workspace_id,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == files_scanned
        assert row[1] == chunks_created
        assert row[2] == duration_ms


class TestCollectionManagement:
    """Tests for collection creation and retrieval."""

    def test_get_or_create_collection_creates_new(self, storage):
        """Verify get_or_create_collection creates new collection."""
        workspace_id = "ws1"
        lang = "python"

        collection = storage.get_or_create_collection(workspace_id, lang)

        assert collection is not None
        assert collection.name is not None

    def test_get_or_create_collection_returns_same(self, storage):
        """Verify get_or_create_collection returns same collection."""
        workspace_id = "ws1"
        lang = "python"

        coll1 = storage.get_or_create_collection(workspace_id, lang)
        coll2 = storage.get_or_create_collection(workspace_id, lang)

        assert coll1.name == coll2.name

    def test_multiple_languages_per_workspace(self, storage):
        """Verify workspace can have multiple language collections."""
        workspace_id = "ws1"

        coll_py = storage.get_or_create_collection(workspace_id, "python")
        coll_js = storage.get_or_create_collection(workspace_id, "javascript")

        assert coll_py.name != coll_js.name
