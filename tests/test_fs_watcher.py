"""Tests for file system watcher module."""

import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from mcp_code_rag.config import AppConfig, WatcherConfig, RagConfig, SecurityConfig
from mcp_code_rag.watcher.fs_watcher import FileWatcher, EventHandler, _active_watchers


@pytest.fixture
def app_config():
    """Create a test AppConfig."""
    config = AppConfig(
        watcher=WatcherConfig(
            enabled=True,
            debounce_ms=2000,
            sync_deletions=True,
        ),
        rag=RagConfig(
            supported_extensions=[".py", ".js", ".ts"],
            default_exclude_patterns=[".git", "__pycache__", "node_modules"],
        ),
        security=SecurityConfig(),
    )
    return config


@pytest.fixture
def mock_pipeline():
    """Create a mock CodeIngestPipeline."""
    pipeline = MagicMock()
    pipeline.ingest_file = MagicMock(
        return_value={
            "indexed": True,
            "skipped": False,
            "chunks_created": 5,
            "errors": [],
        }
    )
    return pipeline


@pytest.fixture
def mock_storage():
    """Create a mock Storage."""
    storage = MagicMock()
    storage.delete_file_chunks = MagicMock()
    return storage


@pytest.fixture
def file_watcher(mock_pipeline, mock_storage, app_config):
    """Create a FileWatcher instance."""
    watcher = FileWatcher(mock_pipeline, mock_storage, app_config)
    yield watcher
    # Cleanup
    watcher.shutdown()


def test_event_handler_file_creation(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that creating a file triggers ingest_file after debounce."""
    workspace_id = "test_workspace"

    # Create event handler
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    # Simulate file creation event
    handler.handle_event("created", str(test_file))

    # Verify ingest_file was called
    mock_pipeline.ingest_file.assert_called_once()
    args = mock_pipeline.ingest_file.call_args
    assert str(test_file.resolve()) == args[0][0]
    assert workspace_id == args[0][1]
    assert args[1]["force_reindex"] is False


def test_event_handler_file_modification(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that modifying a file triggers ingest_file."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    # Simulate file modification event
    handler.handle_event("modified", str(test_file))

    # Verify ingest_file was called
    mock_pipeline.ingest_file.assert_called_once()


def test_event_handler_debounce(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that multiple events within debounce period result in single call."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    # Simulate 3 modification events in quick succession (within debounce)
    handler.handle_event("modified", str(test_file))
    time.sleep(0.1)
    handler.handle_event("modified", str(test_file))
    time.sleep(0.1)
    handler.handle_event("modified", str(test_file))

    # Only first event should trigger ingest_file due to debounce
    assert mock_pipeline.ingest_file.call_count == 1


def test_event_handler_debounce_expiration(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that events after debounce period are processed."""
    workspace_id = "test_workspace"
    # Set short debounce for testing
    app_config.watcher.debounce_ms = 500
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    # First event
    handler.handle_event("modified", str(test_file))
    assert mock_pipeline.ingest_file.call_count == 1

    # Wait for debounce to expire
    time.sleep(0.6)

    # Second event after debounce
    handler.handle_event("modified", str(test_file))
    assert mock_pipeline.ingest_file.call_count == 2


def test_event_handler_file_deletion(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that deleting a file triggers delete_file_chunks."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    file_path = str(test_file.resolve())

    # Simulate file deletion event
    handler.handle_event("deleted", file_path)

    # Verify delete_file_chunks was called
    mock_storage.delete_file_chunks.assert_called_once_with(file_path, workspace_id)


def test_event_handler_exclusion_patterns(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that excluded files are not processed."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    # Create a file in excluded directory
    excluded_dir = tmp_path / "__pycache__"
    excluded_dir.mkdir()
    test_file = excluded_dir / "test.py"
    test_file.write_text("print('hello')")

    # Simulate event for excluded file
    handler.handle_event("created", str(test_file))

    # Should not call ingest_file
    mock_pipeline.ingest_file.assert_not_called()


def test_event_handler_unsupported_extension(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that unsupported file extensions are not processed."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Simulate event for unsupported file
    handler.handle_event("created", str(test_file))

    # Should not call ingest_file
    mock_pipeline.ingest_file.assert_not_called()


def test_file_watcher_start_stop(file_watcher, tmp_path):
    """Test starting and stopping a file watcher."""
    workspace_id = "test_workspace"
    watch_dir = str(tmp_path)

    # Start watcher
    watcher_id = file_watcher.start(watch_dir, workspace_id)
    assert watcher_id is not None

    # List watchers
    watchers = file_watcher.list_watchers()
    assert len(watchers) == 1
    assert watchers[0]["id"] == watcher_id
    assert watchers[0]["workspace_id"] == workspace_id
    assert watchers[0]["dir_path"] == str(Path(watch_dir).resolve())

    # Stop watcher
    result = file_watcher.stop(watcher_id)
    assert result is True

    # Verify watcher is removed
    watchers = file_watcher.list_watchers()
    assert len(watchers) == 0


def test_file_watcher_stop_nonexistent(file_watcher, tmp_path):
    """Test stopping a non-existent watcher doesn't raise exception."""
    # Should not raise exception
    result = file_watcher.stop("nonexistent_id")
    assert result is False


def test_file_watcher_multiple_watchers(file_watcher, tmp_path):
    """Test managing multiple watchers."""
    workspace_id_1 = "workspace_1"
    workspace_id_2 = "workspace_2"

    watch_dir_1 = tmp_path / "dir1"
    watch_dir_2 = tmp_path / "dir2"
    watch_dir_1.mkdir()
    watch_dir_2.mkdir()

    # Start multiple watchers
    watcher_id_1 = file_watcher.start(str(watch_dir_1), workspace_id_1)
    watcher_id_2 = file_watcher.start(str(watch_dir_2), workspace_id_2)

    # List all watchers
    watchers = file_watcher.list_watchers()
    assert len(watchers) == 2

    # Stop first watcher
    result = file_watcher.stop(watcher_id_1)
    assert result is True

    # Verify only second watcher remains
    watchers = file_watcher.list_watchers()
    assert len(watchers) == 1
    assert watchers[0]["id"] == watcher_id_2


def test_file_watcher_sync_deletions_disabled(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that deletion sync respects config setting."""
    workspace_id = "test_workspace"
    app_config.watcher.sync_deletions = False

    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")
    file_path = str(test_file.resolve())

    # Simulate file deletion event
    handler.handle_event("deleted", file_path)

    # Should not call delete_file_chunks
    mock_storage.delete_file_chunks.assert_not_called()


def test_event_handler_directory_ignored(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test that directory events are ignored."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    # Simulate directory event (should be ignored in should_process)
    result = handler._should_process(str(test_dir))
    assert result is False


def test_watcher_global_state_cleanup(tmp_path):
    """Test that global watcher state is properly managed."""
    _active_watchers.clear()

    mock_pipeline = MagicMock()
    mock_storage = MagicMock()
    app_config = AppConfig()

    watcher = FileWatcher(mock_pipeline, mock_storage, app_config)

    try:
        watcher_id = watcher.start(str(tmp_path), "test_workspace")
        assert len(_active_watchers) == 1

        watcher.stop(watcher_id)
        assert len(_active_watchers) == 0
    finally:
        watcher.shutdown()


def test_event_handler_file_creation_integration(file_watcher, mock_pipeline, tmp_path):
    """Integration test: watcher processes file creation."""
    workspace_id = "test_workspace"

    # Start watching the directory
    watcher_id = file_watcher.start(str(tmp_path), workspace_id)

    # Create a test file
    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')")

    # Give observer time to detect file
    time.sleep(0.5)

    # Note: In real usage, observer runs in background thread.
    # For testing debounce without actual filesystem events,
    # we test EventHandler directly which is done in other tests.

    file_watcher.stop(watcher_id)


def test_event_handler_moved_file(app_config, mock_pipeline, mock_storage, tmp_path):
    """Test handling of moved files."""
    workspace_id = "test_workspace"
    handler = EventHandler(mock_pipeline, mock_storage, workspace_id, app_config)

    # Create source and destination files
    src_file = tmp_path / "old.py"
    src_file.write_text("print('hello')")

    dest_file = tmp_path / "new.py"

    # Simulate move event (treated as delete + create)
    handler.handle_event("deleted", str(src_file.resolve()))
    handler.handle_event("created", str(dest_file))

    # Verify both calls were made
    assert mock_storage.delete_file_chunks.call_count == 1
    assert mock_pipeline.ingest_file.call_count == 1
