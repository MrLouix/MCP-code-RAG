"""Tests for the FastMCP server tools (server.py)."""

import json
from unittest.mock import Mock

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from mcp_code_rag.chunker import CodeChunk
from mcp_code_rag.config import AppConfig
from mcp_code_rag.hybrid_search import HybridSearch
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.server import create_server
from mcp_code_rag.storage import Storage


@pytest.fixture
def app_config():
    return AppConfig()


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path / "index"))


@pytest.fixture
def mock_ollama():
    client = Mock(spec=OllamaClient)
    client.embed = Mock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    client.list_models = Mock(return_value=[{"name": "nomic-embed-text"}])
    return client


@pytest.fixture
def pipeline(storage, mock_ollama, app_config):
    return CodeIngestPipeline(storage, mock_ollama, app_config)


@pytest.fixture
def search(storage, mock_ollama, app_config):
    return HybridSearch(storage, mock_ollama, app_config.hybrid_search)


@pytest.fixture
def server(app_config, storage, mock_ollama, pipeline, search):
    return create_server(app_config, storage, mock_ollama, pipeline, search)


async def test_index_project_valid_dir(server, tmp_path):
    """index_project with a valid directory ingests files and returns a summary."""
    py_file = tmp_path / "sample.py"
    py_file.write_text(
        "def greet(name: str) -> str:\n"
        "    '''Greet someone.'''\n"
        "    return f'Hello {name}'\n"
    )

    async with Client(server) as client:
        result = await client.call_tool("index_project", {"directory": str(tmp_path)})

    assert not result.is_error
    assert "Indexed" in result.data


async def test_search_code_returns_results(server, storage):
    """search_code returns results when matching data is in the index."""
    chunk = CodeChunk(
        type="function",
        name="greet",
        package="sample",
        language="python",
        file_path="/workspace/sample.py",
        start_line=1,
        end_line=3,
        selection_start=1,
        selection_end=1,
        signature="def greet(name: str) -> str:",
        docstring="Greet someone.",
        code="def greet(name: str) -> str:\n    return f'Hello {name}'",
    )
    storage.store_chunk(chunk, [0.1] * 768, "ws_search_test")

    async with Client(server) as client:
        result = await client.call_tool(
            "search_code",
            {"query": "greet", "workspace_id": "ws_search_test"},
        )

    assert not result.is_error
    assert result.data is not None


async def test_diagnose_has_required_fields(server):
    """diagnose returns JSON with chromadb, sqlite, and ollama status fields."""
    async with Client(server) as client:
        result = await client.call_tool("diagnose", {})

    assert not result.is_error
    report = json.loads(result.data)
    assert "chromadb" in report
    assert "sqlite" in report
    assert "ollama" in report


async def test_delete_project_nonexistent_workspace(server):
    """delete_project raises a clear error for an unknown workspace ID."""
    with pytest.raises(ToolError) as exc_info:
        async with Client(server) as client:
            await client.call_tool(
                "delete_project", {"workspace_id": "nonexistent_ws_000"}
            )

    assert "nonexistent_ws_000" in str(exc_info.value)
