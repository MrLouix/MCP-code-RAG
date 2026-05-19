"""Tests for Phase 3 server tools: watch_directory, reindex_file, and tagging."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastmcp.client import Client

from mcp_code_rag.config import AppConfig
from mcp_code_rag.hybrid_search import HybridSearch
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.server import create_server
from mcp_code_rag.storage import Storage


@pytest.fixture
def app_config():
    """Load configuration."""
    return AppConfig()


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage(temp_index_dir):
    """Create a Storage instance with temporary directory."""
    return Storage(temp_index_dir)


@pytest.fixture
def mock_ollama():
    """Create a mock OllamaClient."""
    client = Mock(spec=OllamaClient)
    client.embed = Mock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    client.list_models = Mock(return_value=[{"name": "nomic-embed-text"}])
    client.generate = Mock(return_value='{"tags": ["framework:fastapi"]}')
    return client


@pytest.fixture
def pipeline(storage, mock_ollama, app_config):
    """Create a CodeIngestPipeline."""
    return CodeIngestPipeline(storage, mock_ollama, app_config)


@pytest.fixture
def search(storage, mock_ollama, app_config):
    """Create a HybridSearch instance."""
    return HybridSearch(storage, mock_ollama, app_config.hybrid_search)


@pytest.fixture
def mcp_server(app_config, storage, mock_ollama, pipeline, search):
    """Create an MCP server instance."""
    return create_server(app_config, storage, mock_ollama, pipeline, search)


class TestWatchDirectory:
    """Test watch_directory tool."""

    async def test_watch_directory_enabled(self, mcp_server, tmp_path):
        """Test starting a file watcher."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "watch_directory",
                {
                    "directory": str(tmp_path),
                    "workspace_id": "test_ws",
                    "recursive": True,
                    "enabled": True,
                    "format": "human",
                },
            )

        assert not result.is_error
        assert "watching" in result.data.lower() or "started" in result.data.lower()

    async def test_watch_directory_json_format(self, mcp_server, tmp_path):
        """Test watch_directory with JSON format."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "watch_directory",
                {
                    "directory": str(tmp_path),
                    "workspace_id": "test_ws",
                    "recursive": True,
                    "enabled": True,
                    "format": "json",
                },
            )

        assert not result.is_error
        data = json.loads(result.data)
        assert data["status"] == "watching"
        assert "watcher_id" in data
        assert data["directory"] == str(tmp_path)
        assert data["workspace_id"] == "test_ws"

    async def test_watch_directory_enabled_false(self, mcp_server):
        """Test stopping a watcher."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "watch_directory",
                {
                    "directory": "/tmp",
                    "workspace_id": "test_ws",
                    "enabled": False,
                    "format": "human",
                },
            )

        assert not result.is_error
        assert "stopped" in result.data.lower()

    async def test_watch_directory_invalid_path(self, mcp_server):
        """Test watch_directory with non-existent directory."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            async with Client(mcp_server) as client:
                await client.call_tool(
                    "watch_directory",
                    {
                        "directory": "/nonexistent/path/xyz",
                        "enabled": True,
                    },
                )


class TestReindexFile:
    """Test reindex_file tool."""

    async def test_reindex_file_python(self, mcp_server, tmp_path):
        """Test re-indexing a Python file."""
        # Create a temporary Python file
        py_file = tmp_path / "sample.py"
        py_file.write_text(
            """
def hello_world():
    '''A simple function.'''
    return "Hello, World!"

class MyClass:
    '''A test class.'''
    def method(self):
        return 42
"""
        )

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "reindex_file",
                {
                    "file_path": str(py_file),
                    "workspace_id": "default",
                    "format": "human",
                },
            )

        assert not result.is_error
        assert "chunks" in result.data.lower() or str(py_file) in result.data

    async def test_reindex_file_json_format(self, mcp_server, tmp_path):
        """Test reindex_file with JSON format."""
        # Create a temporary Python file
        py_file = tmp_path / "test.py"
        py_file.write_text("def foo():\n    return 42\n")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "reindex_file",
                {
                    "file_path": str(py_file),
                    "workspace_id": "default",
                    "format": "json",
                },
            )

        assert not result.is_error
        data = json.loads(result.data)
        assert "file_path" in data
        assert "chunks_created" in data
        assert "indexed" in data

    async def test_reindex_file_invalid_path(self, mcp_server):
        """Test reindex_file with non-existent file."""
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            async with Client(mcp_server) as client:
                await client.call_tool(
                    "reindex_file",
                    {
                        "file_path": "/nonexistent/file.py",
                    },
                )


class TestTaggingIntegration:
    """Test H1/H2 tagging integration with ingest pipeline."""

    def test_h1_tags_in_chunks(self, pipeline, storage, tmp_path):
        """Test that H1 tags are added to chunks."""
        # Create a test file
        test_file = tmp_path / "test_api.py"
        test_file.write_text(
            """
def api_endpoint():
    return {"status": "ok"}
"""
        )

        # Ingest the file
        result = pipeline.ingest_file(str(test_file), "default")
        assert result["indexed"] is True
        assert result["chunks_created"] > 0

        # Check that tags are stored
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM symbol_index WHERE file_path = ?", (str(test_file),))
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) > 0
        for row in rows:
            if row[0]:  # tags column
                tags = json.loads(row[0])
                assert isinstance(tags, list)
                # Should have at least lang tag
                assert any("lang:" in tag for tag in tags)

    def test_test_file_detection(self, pipeline, storage, tmp_path):
        """Test that test files are tagged correctly."""
        test_file = tmp_path / "module_test.py"
        test_file.write_text("def test_something():\n    assert True\n")

        result = pipeline.ingest_file(str(test_file), "default")
        assert result["indexed"] is True

        # Check for test tag
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM symbol_index WHERE file_path = ?", (str(test_file),))
        rows = cursor.fetchall()
        conn.close()

        found_test_tag = False
        for row in rows:
            if row[0]:
                tags = json.loads(row[0])
                if "type:test" in tags:
                    found_test_tag = True
                    break
        assert found_test_tag

    def test_api_layer_detection(self, pipeline, storage, tmp_path):
        """Test that API layer files are tagged."""
        api_dir = tmp_path / "api"
        api_dir.mkdir()
        api_file = api_dir / "routes.py"
        api_file.write_text("def get_user():\n    return {}\n")

        result = pipeline.ingest_file(str(api_file), "default")
        assert result["indexed"] is True

        # Check for api layer tag
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM symbol_index WHERE file_path = ?", (str(api_file),))
        rows = cursor.fetchall()
        conn.close()

        found_api_tag = False
        for row in rows:
            if row[0]:
                tags = json.loads(row[0])
                if "layer:api" in tags:
                    found_api_tag = True
                    break
        assert found_api_tag

    def test_tags_persisted_in_symbol_index(self, pipeline, storage, tmp_path):
        """Test that tags are persisted in SQLite symbol_index."""
        test_file = tmp_path / "module.py"
        test_file.write_text("class MyClass:\n    pass\n")

        result = pipeline.ingest_file(str(test_file), "default")
        assert result["indexed"] is True

        # Check symbol_index for tags
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol_name, tags FROM symbol_index WHERE file_path = ?",
            (str(test_file),),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) > 0
        for symbol_name, tags_json in rows:
            assert tags_json is not None or tags_json == ""
            if tags_json:
                tags = json.loads(tags_json)
                assert isinstance(tags, list)
                assert len(tags) > 0

    def test_multiple_chunks_same_tags(self, pipeline, storage, tmp_path):
        """Test that all chunks from a file share the same layer tags."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "user.py"
        model_file.write_text(
            """
class User:
    def __init__(self, name):
        self.name = name

def get_user_by_id(user_id):
    return User("test")
"""
        )

        result = pipeline.ingest_file(str(model_file), "default")
        assert result["indexed"] is True

        # Get all chunks for this file
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol_name, tags FROM symbol_index WHERE file_path = ?",
            (str(model_file),),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) > 1  # Should have multiple symbols

        # All should have model tag from file path
        for symbol_name, tags_json in rows:
            if tags_json:
                tags = json.loads(tags_json)
                assert "layer:model" in tags, f"Symbol {symbol_name} missing layer:model tag"
