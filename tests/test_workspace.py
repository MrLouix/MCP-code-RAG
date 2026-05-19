"""Tests for workspace detection and management."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mcp_code_rag.workspace import (
    WorkspaceInfo,
    cache_workspace,
    detect_workspace,
    get_cached_workspace,
    get_collection_name,
    get_workspace_id,
    list_workspaces,
)


class TestDetectWorkspace:
    """Tests for detect_workspace function."""

    def test_detect_workspace_from_root_with_pyproject(self):
        """Test detecting workspace from root directory with pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "pyproject.toml").touch()

            result = detect_workspace(tmpdir)
            assert result is not None
            assert result.name == tmpdir_path.name
            assert result.root == str(tmpdir_path)
            assert result.marker_type == "pyproject.toml"
            assert "python" in result.languages

    def test_detect_workspace_from_subdirectory(self):
        """Test detecting workspace from subdirectory of Python project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "pyproject.toml").touch()

            subdir = tmpdir_path / "src" / "mypackage"
            subdir.mkdir(parents=True)

            result = detect_workspace(str(subdir))
            assert result is not None
            assert result.root == str(tmpdir_path)
            assert result.marker_type == "pyproject.toml"

    def test_detect_workspace_package_json(self):
        """Test detecting JavaScript/TypeScript workspace with package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "package.json").touch()

            result = detect_workspace(tmpdir)
            assert result is not None
            assert result.marker_type == "package.json"
            assert "javascript" in result.languages
            assert "typescript" in result.languages

    def test_detect_workspace_cargo_toml(self):
        """Test detecting Rust workspace with Cargo.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "Cargo.toml").touch()

            result = detect_workspace(tmpdir)
            assert result is not None
            assert result.marker_type == "Cargo.toml"
            assert result.languages == ["rust"]

    def test_detect_workspace_marker_priority(self):
        """Test that markers are checked in correct priority order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create multiple markers - pyproject.toml should win
            (tmpdir_path / "pyproject.toml").touch()
            (tmpdir_path / "package.json").touch()

            result = detect_workspace(tmpdir)
            assert result.marker_type == "pyproject.toml"

    def test_detect_workspace_no_marker_returns_none(self):
        """Test that detect_workspace returns None when no markers found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = detect_workspace(tmpdir)
            assert result is None

    def test_detect_workspace_empty_subdirectory_returns_none(self):
        """Test that detect_workspace returns None from empty subdirectory without markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()

            result = detect_workspace(str(subdir))
            assert result is None

    def test_detect_workspace_go_mod(self):
        """Test detecting Go workspace via go.mod marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "go.mod").touch()

            result = detect_workspace(tmpdir)
            assert result is not None
            assert result.marker_type == "go.mod"
            assert "go" in result.languages

    def test_detect_workspace_deep_nesting(self):
        """Test detecting workspace root from a deeply nested subdirectory (4+ levels)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "pyproject.toml").touch()

            deep = tmpdir_path / "src" / "pkg" / "module" / "submodule"
            deep.mkdir(parents=True)

            result = detect_workspace(str(deep))
            assert result is not None
            assert result.root == str(tmpdir_path)
            assert result.marker_type == "pyproject.toml"

    def test_detect_workspace_gemfile(self):
        """Test detecting Ruby workspace via Gemfile marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "Gemfile").touch()

            result = detect_workspace(tmpdir)
            assert result is not None
            assert result.marker_type == "Gemfile"
            assert "ruby" in result.languages


class TestGetWorkspaceId:
    """Tests for get_workspace_id function."""

    def test_get_workspace_id_exact_length(self):
        """Test that get_workspace_id returns exactly 12 characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_id = get_workspace_id(tmpdir)
            assert len(workspace_id) == 12

    def test_get_workspace_id_is_hex(self):
        """Test that get_workspace_id returns hexadecimal characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_id = get_workspace_id(tmpdir)
            assert all(c in "0123456789abcdef" for c in workspace_id)

    def test_get_workspace_id_stable_same_path(self):
        """Test that same path always produces same ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            id1 = get_workspace_id(tmpdir)
            id2 = get_workspace_id(tmpdir)
            assert id1 == id2

    def test_get_workspace_id_different_paths_different_ids(self):
        """Test that different paths produce different IDs."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                id1 = get_workspace_id(tmpdir1)
                id2 = get_workspace_id(tmpdir2)
                assert id1 != id2

    def test_get_workspace_id_resolves_relative_path(self):
        """Test that relative paths are resolved to absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create a marker file so we can cd into it
            (tmpdir_path / "pyproject.toml").touch()

            # Get ID from absolute path
            abs_id = get_workspace_id(str(tmpdir_path))

            # Get ID from relative path (from tmpdir)
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir_path)
                rel_id = get_workspace_id(".")
                assert abs_id == rel_id
            finally:
                os.chdir(old_cwd)


class TestGetCollectionName:
    """Tests for get_collection_name function."""

    def test_get_collection_name_format(self):
        """Test that collection name has correct format."""
        name = get_collection_name("abc123def456", "python")
        assert name == "coderag-abc123def456-python"

    def test_get_collection_name_with_different_languages(self):
        """Test collection names for various languages."""
        workspace_id = "workspace123456"

        for lang in ["python", "javascript", "typescript", "go", "rust"]:
            name = get_collection_name(workspace_id, lang)
            assert name == f"coderag-{workspace_id}-{lang}"

    def test_get_collection_name_with_short_id(self):
        """Test collection name with short workspace ID."""
        name = get_collection_name("abc123", "java")
        assert name == "coderag-abc123-java"


class TestWorkspaceCache:
    """Tests for workspace caching functions."""

    def test_cache_workspace_creates_table(self):
        """Test that cache_workspace creates necessary tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            workspace_info = WorkspaceInfo(
                id="test123abc456",
                name="test_project",
                root="/tmp/test",
                marker_type="pyproject.toml",
                languages=["python"],
            )

            cache_workspace(str(db_path), workspace_info)

            # Verify table exists and has data
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM workspace_cache")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 1

    def test_cache_and_get_workspace_roundtrip(self):
        """Test caching and retrieving workspace information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            original = WorkspaceInfo(
                id="abc123def456",
                name="my_workspace",
                root="/home/user/projects/myproject",
                marker_type="pyproject.toml",
                languages=["python"],
            )

            cache_workspace(str(db_path), original)
            retrieved = get_cached_workspace(str(db_path), original.id)

            assert retrieved is not None
            assert retrieved.id == original.id
            assert retrieved.name == original.name
            assert retrieved.root == original.root
            assert retrieved.marker_type == original.marker_type
            assert retrieved.languages == original.languages

    def test_get_cached_workspace_not_found(self):
        """Test that get_cached_workspace returns None for missing workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            result = get_cached_workspace(str(db_path), "nonexistent_id")
            assert result is None

    def test_cache_workspace_multiple_languages(self):
        """Test caching workspace with multiple languages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            workspace = WorkspaceInfo(
                id="xyz789abc123",
                name="polyglot_project",
                root="/home/user/polyglot",
                marker_type="package.json",
                languages=["javascript", "typescript"],
            )

            cache_workspace(str(db_path), workspace)
            retrieved = get_cached_workspace(str(db_path), workspace.id)

            assert retrieved is not None
            assert len(retrieved.languages) == 2
            assert "javascript" in retrieved.languages
            assert "typescript" in retrieved.languages

    def test_cache_workspace_replace(self):
        """Test that caching with same ID replaces old entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            workspace1 = WorkspaceInfo(
                id="same_id_123456",
                name="old_name",
                root="/old/path",
                marker_type="pyproject.toml",
                languages=["python"],
            )

            workspace2 = WorkspaceInfo(
                id="same_id_123456",
                name="new_name",
                root="/new/path",
                marker_type="setup.py",
                languages=["python"],
            )

            cache_workspace(str(db_path), workspace1)
            cache_workspace(str(db_path), workspace2)

            retrieved = get_cached_workspace(str(db_path), "same_id_123456")
            assert retrieved.name == "new_name"
            assert retrieved.root == "/new/path"


class TestListWorkspaces:
    """Tests for list_workspaces function."""

    def test_list_workspaces_empty_database(self):
        """Test list_workspaces on empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            result = list_workspaces(str(db_path))
            assert result == []

    def test_list_workspaces_multiple_entries(self):
        """Test list_workspaces returns all cached workspaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            workspaces = [
                WorkspaceInfo(
                    id="id001abc123456",
                    name="project1",
                    root="/home/user/project1",
                    marker_type="pyproject.toml",
                    languages=["python"],
                ),
                WorkspaceInfo(
                    id="id002def789012",
                    name="project2",
                    root="/home/user/project2",
                    marker_type="package.json",
                    languages=["javascript", "typescript"],
                ),
                WorkspaceInfo(
                    id="id003ghi345678",
                    name="project3",
                    root="/home/user/project3",
                    marker_type="Cargo.toml",
                    languages=["rust"],
                ),
            ]

            for workspace in workspaces:
                cache_workspace(str(db_path), workspace)

            result = list_workspaces(str(db_path))
            assert len(result) == 3

            # Check that all workspaces are in result
            result_ids = {w.id for w in result}
            expected_ids = {w.id for w in workspaces}
            assert result_ids == expected_ids

    def test_list_workspaces_ordered_by_creation(self):
        """Test that list_workspaces returns results ordered by creation time (most recent first)."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            workspace1 = WorkspaceInfo(
                id="first_id_12345",
                name="first",
                root="/first",
                marker_type="pyproject.toml",
                languages=["python"],
            )

            workspace2 = WorkspaceInfo(
                id="second_id_1234",
                name="second",
                root="/second",
                marker_type="package.json",
                languages=["javascript"],
            )

            cache_workspace(str(db_path), workspace1)
            time.sleep(1)  # Delay to ensure different timestamps
            cache_workspace(str(db_path), workspace2)

            result = list_workspaces(str(db_path))
            # Should be in reverse order (most recent first)
            assert len(result) == 2
            assert result[0].id == "second_id_1234"
            assert result[1].id == "first_id_12345"
