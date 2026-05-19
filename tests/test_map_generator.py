"""Tests for map_generator module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mcp_code_rag.chunker import CodeChunk
from mcp_code_rag.map_generator import MapGenerator
from mcp_code_rag.storage import Storage


@pytest.fixture
def temp_storage():
    """Create a temporary storage instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        yield storage
        storage.close()


@pytest.fixture
def sample_data(temp_storage):
    """Populate storage with sample data."""
    workspace_id = "test_workspace"

    # Create sample chunks
    chunks = [
        CodeChunk(
            type="function",
            name="process_data",
            package="data_handler",
            language="python",
            file_path="/workspace/data_handler.py",
            start_line=10,
            end_line=25,
            selection_start=10,
            selection_end=10,
            signature="def process_data(data: dict) -> None",
            docstring="Process input data",
            code="def process_data(data):\n    pass",
        ),
        CodeChunk(
            type="class",
            name="DataProcessor",
            package="data_handler",
            language="python",
            file_path="/workspace/data_handler.py",
            start_line=30,
            end_line=50,
            selection_start=30,
            selection_end=30,
            signature="class DataProcessor",
            docstring="Data processor class",
            code="class DataProcessor:\n    pass",
        ),
        CodeChunk(
            type="method",
            name="_private_method",
            package="data_handler.DataProcessor",
            language="python",
            file_path="/workspace/data_handler.py",
            start_line=35,
            end_line=45,
            selection_start=35,
            selection_end=35,
            signature="def _private_method(self)",
            code="def _private_method(self):\n    pass",
        ),
        CodeChunk(
            type="function",
            name="validate_input",
            package="validators",
            language="python",
            file_path="/workspace/validators.py",
            start_line=1,
            end_line=10,
            selection_start=1,
            selection_end=1,
            signature="def validate_input(x: str) -> bool",
            code="def validate_input(x):\n    return True",
        ),
        CodeChunk(
            type="config",
            name="config.json",
            package="",
            language="config",
            file_path="/workspace/config/config.json",
            start_line=1,
            end_line=20,
            selection_start=1,
            selection_end=1,
            signature="config.json",
            code='{"version": "1.0"}',
        ),
    ]

    # Store chunks with embeddings
    import numpy as np

    for chunk in chunks:
        embedding = np.random.rand(768).tolist()
        temp_storage.store_chunk(chunk, embedding, workspace_id)

    # Record scan history
    temp_storage.record_scan(workspace_id, 3, 5, 1000)
    temp_storage.record_scan(workspace_id, 4, 7, 1200)

    return workspace_id, temp_storage


def test_generate_project_map_summary(sample_data):
    """Test generate_project_map with summary format."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    result = generator.generate_project_map(workspace_id, format="summary")

    assert "summary" in result
    summary = result["summary"]

    # Check required keys
    assert "total_files" in summary
    assert "total_lines" in summary
    assert "stats_by_language" in summary
    assert "top_level_folders" in summary
    assert "most_active_files" in summary
    assert "last_indexed" in summary

    # Check values
    assert summary["total_files"] == 3
    assert summary["total_lines"] > 0
    assert "python" in summary["stats_by_language"]
    assert "config" in summary["stats_by_language"]


def test_generate_project_map_detailed(sample_data):
    """Test generate_project_map with detailed format."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    result = generator.generate_project_map(workspace_id, format="detailed")

    assert "summary" in result
    assert "detailed" in result
    assert "files" in result["detailed"]
    assert len(result["detailed"]["files"]) == 3

    # Check file structure
    for file_info in result["detailed"]["files"]:
        assert "path" in file_info
        assert "language" in file_info
        assert "functions" in file_info
        assert "classes" in file_info
        assert "methods" in file_info


def test_generate_file_tree(sample_data):
    """Test generate_file_tree returns valid tree structure."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    tree = generator.generate_file_tree(workspace_id)

    # Check that tree contains expected elements
    assert isinstance(tree, str)
    assert len(tree) > 0

    # Check for tree characters
    assert any(char in tree for char in ["├", "└", "│"])

    # Check for file names
    assert "data_handler.py" in tree
    assert "validators.py" in tree
    assert "config.json" in tree


def test_generate_file_tree_with_filter(sample_data):
    """Test generate_file_tree with path filter."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    tree = generator.generate_file_tree(workspace_id, path_filter="*/data_handler.py")

    assert "data_handler.py" in tree


def test_generate_project_stats(sample_data):
    """Test generate_project_stats includes delta between scans."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    stats = generator.generate_project_stats(workspace_id)

    assert "current" in stats
    assert "scan_history" in stats
    assert "delta" in stats

    # Check current stats
    assert "files" in stats["current"]
    assert "total_lines" in stats["current"]
    assert "total_symbols" in stats["current"]

    # Check scan history
    assert len(stats["scan_history"]) >= 2

    # Check delta (difference between first and last scan)
    delta = stats["delta"]
    assert "files_delta" in delta
    assert "chunks_delta" in delta
    assert "time_span_ms" in delta

    # Verify delta calculation
    first = stats["scan_history"][0]
    last = stats["scan_history"][-1]
    assert delta["files_delta"] == last["files_scanned"] - first["files_scanned"]
    assert delta["chunks_delta"] == last["chunks_created"] - first["chunks_created"]


def test_list_package_exports(sample_data):
    """Test list_package_exports excludes underscore-prefixed symbols."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    # Query symbols from data_handler package
    exports = generator.list_package_exports(workspace_id, "data_handler")

    # Should include process_data and DataProcessor
    assert "process_data" in exports
    assert "DataProcessor" in exports

    # Should NOT include _private_method
    assert "_private_method" not in exports


def test_list_package_exports_validators(sample_data):
    """Test list_package_exports for validators package."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    exports = generator.list_package_exports(workspace_id, "validators")

    assert "validate_input" in exports
    assert len(exports) >= 1


def test_generate_project_map_tree_format(sample_data):
    """Test generate_project_map with tree format."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    result = generator.generate_project_map(workspace_id, format="tree")

    assert "tree" in result
    tree = result["tree"]
    assert isinstance(tree, str)
    assert len(tree) > 0
    assert "data_handler.py" in tree or "validators.py" in tree


def test_infer_language(sample_data):
    """Test _infer_language helper."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    assert generator._infer_language("/workspace/script.py") == "python"
    assert generator._infer_language("/workspace/main.go") == "go"
    assert generator._infer_language("/workspace/index.js") == "javascript"
    assert generator._infer_language("/workspace/unknown.xyz") == "unknown"


def test_get_top_level_folders(sample_data):
    """Test _get_top_level_folders helper."""
    workspace_id, storage = sample_data
    generator = MapGenerator(storage)

    files = [
        "/workspace/data_handler.py",
        "/workspace/validators.py",
        "/workspace/config/config.json",
    ]

    folders = generator._get_top_level_folders(files)

    assert "workspace" in folders
    assert "config" not in folders  # config is not at top level of all paths
