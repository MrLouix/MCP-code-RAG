"""Tests for the code ingestion pipeline."""

import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from mcp_code_rag.config import AppConfig
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.storage import Storage
from mcp_code_rag.ollama_client import OllamaClient


@pytest.fixture
def app_config():
    """Create a test AppConfig with reasonable defaults."""
    config = AppConfig()
    # Reduce max file size for testing
    config.security.max_file_size_mb = 1.0
    return config


@pytest.fixture
def storage(tmp_path):
    """Create a test Storage instance."""
    return Storage(str(tmp_path / "index"))


@pytest.fixture
def ollama_client(monkeypatch):
    """Create a mock OllamaClient."""
    client = Mock(spec=OllamaClient)
    # Mock embed to return correct number of embeddings based on input
    def mock_embed(texts):
        return [[0.1] * 768 for _ in texts]
    client.embed = Mock(side_effect=mock_embed)
    return client


@pytest.fixture
def pipeline(storage, ollama_client, app_config):
    """Create a CodeIngestPipeline instance."""
    return CodeIngestPipeline(storage, ollama_client, app_config)


class TestIngestFile:
    """Tests for CodeIngestPipeline.ingest_file()."""

    def test_ingest_simple_python_file(self, pipeline, tmp_path, ollama_client):
        """Test ingesting a simple Python file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    '''Say hello.'''
    return "Hello, World!"

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
""")

        workspace_id = "test-workspace"

        # Mock embedding to return correct number of vectors
        ollama_client.embed.return_value = [[0.1] * 768 for _ in range(10)]

        result = pipeline.ingest_file(str(test_file), workspace_id)

        assert result["indexed"] is True
        assert result["skipped"] is False
        assert result["chunks_created"] > 0
        assert len(result["errors"]) == 0 or "dependency" in str(result["errors"]).lower()

    def test_ingest_file_hash_cache_skip(self, pipeline, tmp_path, ollama_client):
        """Test that identical files are skipped on second ingest."""
        test_file = tmp_path / "cached.py"
        test_file.write_text("def func(): pass")

        workspace_id = "test-workspace"
        ollama_client.embed.return_value = [[0.1] * 768]

        # First ingest
        result1 = pipeline.ingest_file(str(test_file), workspace_id)
        assert result1["indexed"] is True

        # Second ingest (should be skipped)
        result2 = pipeline.ingest_file(str(test_file), workspace_id)
        assert result2["skipped"] is True
        assert result2["indexed"] is False

    def test_ingest_file_force_reindex(self, pipeline, tmp_path, ollama_client):
        """Test that force_reindex ignores hash cache."""
        test_file = tmp_path / "force.py"
        test_file.write_text("def func(): pass")

        workspace_id = "test-workspace"
        ollama_client.embed.return_value = [[0.1] * 768]

        # First ingest
        result1 = pipeline.ingest_file(str(test_file), workspace_id)
        assert result1["indexed"] is True

        # Second ingest with force_reindex
        result2 = pipeline.ingest_file(str(test_file), workspace_id, force_reindex=True)
        assert result2["indexed"] is True
        assert result2["skipped"] is False

    def test_ingest_file_exceeds_size_limit(self, pipeline, tmp_path, app_config):
        """Test that files exceeding max_file_size_mb are skipped."""
        # Create a large file (larger than 1 MB limit from fixture)
        large_file = tmp_path / "large.py"
        large_content = "x = 1\n" * 200000  # ~2 MB
        large_file.write_text(large_content)

        workspace_id = "test-workspace"

        result = pipeline.ingest_file(str(large_file), workspace_id)

        assert result["skipped"] is True
        assert result["indexed"] is False

    def test_ingest_unsupported_file_type(self, pipeline, tmp_path):
        """Test that unsupported file types are skipped."""
        unsupported_file = tmp_path / "data.xyz"
        unsupported_file.write_text("some binary data")

        workspace_id = "test-workspace"

        result = pipeline.ingest_file(str(unsupported_file), workspace_id)

        assert result["skipped"] is True

    def test_ingest_modified_file_reindexed(self, pipeline, tmp_path, ollama_client):
        """Test that modified files are re-indexed."""
        test_file = tmp_path / "modified.py"
        test_file.write_text("def func1(): pass")

        workspace_id = "test-workspace"
        ollama_client.embed.return_value = [[0.1] * 768]

        # First ingest
        result1 = pipeline.ingest_file(str(test_file), workspace_id)
        assert result1["indexed"] is True

        # Modify file
        time.sleep(0.01)  # Ensure different timestamp/content
        test_file.write_text("def func1(): pass\ndef func2(): pass")

        # Second ingest (should re-index due to hash change)
        result2 = pipeline.ingest_file(str(test_file), workspace_id)
        assert result2["indexed"] is True
        assert result2["skipped"] is False


class TestIngestDirectory:
    """Tests for CodeIngestPipeline.ingest_directory()."""

    def test_ingest_directory_two_files(self, pipeline, tmp_path, ollama_client):
        """Test ingesting a directory with two Python files."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create two test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def func2(): pass")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        assert result["files_scanned"] == 3  # 2 files + 1 marker
        assert result["files_indexed"] == 3  # 2 py files + 1 config
        assert result["files_skipped"] == 0
        assert result["total_chunks"] > 0
        assert "python" in result["languages"]
        assert "config" in result["languages"]
        assert len(result["errors"]) == 0

    def test_ingest_directory_second_run_skips_unchanged(self, pipeline, tmp_path, ollama_client):
        """Test that second run skips unchanged files."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def func2(): pass")

        # First run
        result1 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result1["files_indexed"] == 3  # 2 py files + 1 config
        assert result1["files_skipped"] == 0

        # Second run (should skip all files)
        result2 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result2["files_indexed"] == 0
        assert result2["files_skipped"] == 3  # All 3 files skipped

    def test_ingest_directory_modify_one_file(self, pipeline, tmp_path, ollama_client):
        """Test that modifying one file re-indexes only that file."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def func2(): pass")

        # First run
        result1 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result1["files_indexed"] == 3  # 2 py + 1 config

        # Modify one file
        time.sleep(0.01)
        file1.write_text("def func1(): pass\ndef new_func(): pass")

        # Second run
        result2 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result2["files_indexed"] == 1  # Only file1 re-indexed
        assert result2["files_skipped"] == 2  # file2 + pyproject.toml skipped

    def test_ingest_directory_excludes_pycache(self, pipeline, tmp_path, ollama_client):
        """Test that __pycache__ is excluded from indexing."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create a regular file
        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        # Create __pycache__ directory with a file
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        cache_file = pycache / "file1.cpython-39.pyc"
        cache_file.write_text("fake compiled python")

        result = pipeline.ingest_directory(str(tmp_path), recursive=True)

        # Python file + pyproject.toml should be indexed
        assert result["files_indexed"] == 2  # file1.py + pyproject.toml
        # __pycache__ files should be excluded
        assert result["files_scanned"] >= 2  # file1.py and cache_file
        # cache_file should be skipped due to pattern exclusion
        assert result["files_skipped"] >= 1

    def test_ingest_directory_skips_large_files(self, pipeline, tmp_path, app_config, ollama_client):
        """Test that files exceeding max_file_size_mb are skipped."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create a normal file
        normal_file = tmp_path / "normal.py"
        normal_file.write_text("def func(): pass")

        # Create a large file
        large_file = tmp_path / "large.py"
        large_file.write_text("x = 1\n" * 200000)  # ~2 MB

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        assert result["files_scanned"] == 3  # normal, large, pyproject.toml
        assert result["files_indexed"] == 2  # normal file + pyproject.toml config
        assert result["files_skipped"] == 1  # Large file skipped

    def test_ingest_directory_recursive(self, pipeline, tmp_path, ollama_client):
        """Test recursive directory traversal."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = subdir / "file2.py"
        file2.write_text("def func2(): pass")

        result = pipeline.ingest_directory(str(tmp_path), recursive=True)

        assert result["files_indexed"] == 3  # file1.py + file2.py + pyproject.toml

    def test_ingest_directory_non_recursive(self, pipeline, tmp_path, ollama_client):
        """Test non-recursive directory traversal."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = subdir / "file2.py"
        file2.write_text("def func2(): pass")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Should only index file1 and pyproject.toml, not file2 in subdir
        assert result["files_indexed"] == 2  # file1.py + pyproject.toml

    def test_ingest_directory_returns_language_stats(self, pipeline, tmp_path, ollama_client):
        """Test that ingest_directory returns per-language statistics."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create files in different languages
        py_file = tmp_path / "script.py"
        py_file.write_text("def func(): pass")

        js_file = tmp_path / "script.js"
        js_file.write_text("function func() {}")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        assert result["languages"]["python"] >= 1
        assert result["languages"]["javascript"] >= 1

    def test_ingest_directory_records_scan_history(self, pipeline, tmp_path, ollama_client):
        """Test that scan events are recorded in scan_history."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        file1 = tmp_path / "file1.py"
        file1.write_text("def func(): pass")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Verify that duration_ms is recorded
        assert result["duration_ms"] > 0
        assert result["duration_ms"] < 60000  # Less than 60 seconds

    def test_ingest_directory_with_custom_exclude_patterns(self, pipeline, tmp_path, ollama_client):
        """Test that custom exclude patterns are respected."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create files
        file1 = tmp_path / "keep.py"
        file1.write_text("def func(): pass")

        file2 = tmp_path / "exclude_me.py"
        file2.write_text("def func(): pass")

        result = pipeline.ingest_directory(
            str(tmp_path),
            recursive=False,
            exclude_patterns=["exclude_*"],
        )

        # keep.py + pyproject.toml should be indexed
        assert result["files_indexed"] == 2
        assert result["files_skipped"] == 1  # exclude_me.py

    def test_ingest_directory_error_handling(self, pipeline, tmp_path, ollama_client):
        """Test that errors are collected and reported."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        file1 = tmp_path / "file1.py"
        file1.write_text("def func(): pass")

        # Mock embed to raise an error
        ollama_client.embed.side_effect = Exception("Embedding service error")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Should record errors but continue
        assert len(result["errors"]) > 0
        assert "Embedding service error" in str(result["errors"])

    def test_ingest_directory_handles_invalid_python(self, pipeline, tmp_path, ollama_client):
        """Test that files with syntax errors are handled gracefully."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Create a file with syntax error
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def func(:\n    pass")  # Missing closing paren

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Should have scanned files
        assert result["files_scanned"] == 2  # bad.py + pyproject.toml
        # The pyproject.toml should be indexed, bad.py should be skipped due to parse error
        assert result["files_indexed"] == 1  # Only pyproject.toml
        assert result["files_skipped"] == 1  # bad.py

    def test_ingest_directory_with_mixed_extensions(self, pipeline, tmp_path, ollama_client):
        """Test directory with mixed file types."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Supported files
        py_file = tmp_path / "script.py"
        py_file.write_text("def func(): pass")

        md_file = tmp_path / "README.md"
        md_file.write_text("# Documentation")

        # Unsupported file
        txt_file = tmp_path / "data.xyz"
        txt_file.write_text("some data")

        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Should index .py, .md, and .toml, skip .xyz
        assert result["files_indexed"] == 3  # .py + .md + .toml
        assert result["files_scanned"] == 4
        assert result["files_skipped"] == 1  # .xyz
        assert "python" in result["languages"]
        assert "doc" in result["languages"]
        assert "config" in result["languages"]


class TestIngestPipelineIntegration:
    """Integration tests for the full ingestion workflow."""

    def test_full_workflow_single_run(self, pipeline, tmp_path, ollama_client):
        """Test complete workflow: create files, ingest, verify results."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Setup
        py1 = tmp_path / "module1.py"
        py1.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
""")

        py2 = tmp_path / "module2.py"
        py2.write_text("""
def main():
    pass
""")

        # Execute
        result = pipeline.ingest_directory(str(tmp_path), recursive=False)

        # Verify
        assert result["files_scanned"] == 3  # 2 py files + 1 marker
        assert result["files_indexed"] == 3  # 2 py + 1 config
        assert result["files_skipped"] == 0
        assert result["total_chunks"] > 2
        assert result["duration_ms"] > 0
        assert len(result["errors"]) == 0

    def test_incremental_updates(self, pipeline, tmp_path, ollama_client):
        """Test incremental updates over multiple ingest runs."""
        # Create workspace marker
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        # Run 1: Create 2 files
        file1 = tmp_path / "file1.py"
        file1.write_text("def f1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def f2(): pass")

        result1 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result1["files_indexed"] == 3  # 2 py + 1 config

        # Run 2: No changes
        result2 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result2["files_indexed"] == 0
        assert result2["files_skipped"] == 3  # All 3 files skipped

        # Run 3: Add one file, modify one file
        file1.write_text("def f1(): return 1")  # Modify
        file3 = tmp_path / "file3.py"
        file3.write_text("def f3(): pass")  # New

        result3 = pipeline.ingest_directory(str(tmp_path), recursive=False)
        assert result3["files_indexed"] == 2  # file1 (modified) + file3 (new)
        assert result3["files_skipped"] == 2  # file2 + marker
