"""File system watcher for real-time code indexing using watchdog."""

import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

from watchdog.events import FileModifiedEvent, FileCreatedEvent, FileDeletedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from mcp_code_rag.config import AppConfig
from mcp_code_rag.extractors import is_excluded
from mcp_code_rag.ingest import CodeIngestPipeline
from mcp_code_rag.storage import Storage

logger = logging.getLogger(__name__)

_active_watchers = {}


class EventHandler(FileSystemEventHandler):
    """Handles file system events with debouncing and exclusion filtering."""

    def __init__(
        self,
        pipeline: CodeIngestPipeline,
        storage: Storage,
        workspace_id: str,
        config: AppConfig,
    ):
        """
        Initialize event handler.

        Args:
            pipeline: CodeIngestPipeline instance.
            storage: Storage instance.
            workspace_id: Workspace identifier.
            config: Application configuration.
        """
        super().__init__()
        self.pipeline = pipeline
        self.storage = storage
        self.workspace_id = workspace_id
        self.config = config
        self.debounce_ms = config.watcher.debounce_ms
        self.sync_deletions = config.watcher.sync_deletions
        self.exclude_patterns = config.rag.default_exclude_patterns
        self.supported_extensions = config.rag.supported_extensions

        # Track last event time per file for debouncing
        self._last_event_time = {}

    def _check_debounce(self, file_path: str) -> bool:
        """Check if enough time has passed since last event for this file."""
        current_time = time.time() * 1000  # Convert to ms
        last_time = self._last_event_time.get(file_path, 0)
        if current_time - last_time < self.debounce_ms:
            return False
        self._last_event_time[file_path] = current_time
        return True

    def _should_process(self, file_path: str) -> bool:
        """Check if file should be processed based on debounce and exclusions."""
        # Skip directories
        if Path(file_path).is_dir():
            return False

        # Check exclusion patterns
        if is_excluded(file_path, self.exclude_patterns):
            return False

        # Check file extension
        if Path(file_path).suffix not in self.supported_extensions:
            return False

        # Check debounce
        if not self._check_debounce(file_path):
            return False

        return True

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            self.handle_event("created", event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification."""
        if event.is_directory:
            return

        if self._should_process(event.src_path):
            self.handle_event("modified", event.src_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion."""
        if event.is_directory:
            return

        file_path = event.src_path

        # Skip exclusion check for deletions (file already gone)
        if Path(file_path).suffix not in self.supported_extensions:
            return

        self.handle_event("deleted", file_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle file move (treat as delete + create)."""
        if event.is_directory:
            return

        src_path = event.src_path
        dest_path = event.dest_path

        # Only process if destination is not excluded
        if is_excluded(dest_path, self.exclude_patterns):
            # Just delete the source
            if Path(src_path).suffix in self.supported_extensions:
                self.handle_event("deleted", src_path)
            return

        # Check extension
        if Path(dest_path).suffix not in self.supported_extensions:
            return

        # Treat as delete + create
        self.handle_event("deleted", src_path)
        self.handle_event("created", dest_path)

    def handle_event(self, event_type: str, file_path: str) -> None:
        """
        Process a file system event.

        Args:
            event_type: Type of event ("created", "modified", "deleted", "moved").
            file_path: Path to the affected file.
        """
        try:
            file_path_abs = str(Path(file_path).resolve())

            # Skip directories
            if Path(file_path_abs).is_dir():
                return

            # Check exclusion patterns and extension for non-deletion events
            if event_type != "deleted":
                if is_excluded(file_path_abs, self.exclude_patterns):
                    return
                if Path(file_path_abs).suffix not in self.supported_extensions:
                    return
                # Apply debounce for create/modify events
                if not self._check_debounce(file_path_abs):
                    return

            if event_type in ("created", "modified"):
                # Ingest or update the file
                result = self.pipeline.ingest_file(
                    file_path_abs,
                    self.workspace_id,
                    force_reindex=False,
                )
                if result["indexed"]:
                    logger.info(f"Indexed file: {file_path_abs} ({result['chunks_created']} chunks)")
                elif result["skipped"]:
                    logger.debug(f"Skipped file: {file_path_abs}")
                else:
                    logger.warning(f"Failed to index {file_path_abs}: {result['errors']}")

            elif event_type == "deleted":
                # Delete chunks for the file
                if self.sync_deletions:
                    self.storage.delete_file_chunks(file_path_abs, self.workspace_id)
                    logger.info(f"Deleted chunks for: {file_path_abs}")

        except Exception as e:
            logger.error(f"Error handling {event_type} event for {file_path}: {e}")


class FileWatcher:
    """Manages file system watchers for real-time code indexing."""

    def __init__(
        self,
        pipeline: CodeIngestPipeline,
        storage: Storage,
        config: AppConfig,
    ):
        """
        Initialize FileWatcher.

        Args:
            pipeline: CodeIngestPipeline instance.
            storage: Storage instance.
            config: Application configuration.
        """
        self.pipeline = pipeline
        self.storage = storage
        self.config = config
        self.observer = Observer()
        self.observer.start()

    def start(
        self,
        dir_path: str,
        workspace_id: str,
        recursive: bool = True,
    ) -> str:
        """
        Start watching a directory for changes.

        Args:
            dir_path: Directory path to watch.
            workspace_id: Workspace identifier.
            recursive: If True, watch subdirectories.

        Returns:
            Unique watcher ID.
        """
        dir_path_abs = str(Path(dir_path).resolve())

        # Create event handler
        event_handler = EventHandler(
            self.pipeline,
            self.storage,
            workspace_id,
            self.config,
        )

        # Schedule observer
        self.observer.schedule(event_handler, dir_path_abs, recursive=recursive)

        # Generate unique watcher ID
        watcher_id = str(uuid.uuid4())

        # Store watcher info
        _active_watchers[watcher_id] = {
            "id": watcher_id,
            "dir_path": dir_path_abs,
            "workspace_id": workspace_id,
            "recursive": recursive,
            "event_handler": event_handler,
            "started_at": time.time(),
        }

        logger.info(f"Started watching {dir_path_abs} (ID: {watcher_id})")
        return watcher_id

    def stop(self, watcher_id: str) -> bool:
        """
        Stop watching a directory.

        Args:
            watcher_id: ID of the watcher to stop.

        Returns:
            True if watcher was found and stopped, False otherwise.
        """
        if watcher_id not in _active_watchers:
            return False

        watcher_info = _active_watchers[watcher_id]
        dir_path = watcher_info["dir_path"]

        # Unschedule observer
        self.observer.unschedule_all()

        # Re-schedule remaining watchers
        for w_id, w_info in _active_watchers.items():
            if w_id != watcher_id:
                self.observer.schedule(
                    w_info["event_handler"],
                    w_info["dir_path"],
                    recursive=w_info["recursive"],
                )

        # Remove from active watchers
        del _active_watchers[watcher_id]

        logger.info(f"Stopped watching {dir_path} (ID: {watcher_id})")
        return True

    def list_watchers(self) -> list[dict]:
        """
        List all active watchers.

        Returns:
            List of watcher info dictionaries.
        """
        return [
            {
                "id": w["id"],
                "dir_path": w["dir_path"],
                "workspace_id": w["workspace_id"],
                "recursive": w["recursive"],
                "started_at": w["started_at"],
            }
            for w in _active_watchers.values()
        ]

    def shutdown(self) -> None:
        """Shutdown the file watcher and stop all observers."""
        self.observer.stop()
        self.observer.join()
        _active_watchers.clear()
        logger.info("FileWatcher shutdown complete")
