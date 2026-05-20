"""Ingestion pipeline for indexing source code files into ChromaDB and SQLite."""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from mcp_code_rag.chunker import chunk_file
from mcp_code_rag.config import AppConfig
from mcp_code_rag.extractors import detect_language, extract_imports_python, extract_imports_generic, is_excluded
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.storage import Storage
from mcp_code_rag.tagging.heuristics import tag_h1
from mcp_code_rag.tagging.llm_tagger import LLMTagger
from mcp_code_rag.utils.hashing import hash_file
from mcp_code_rag.workspace import detect_workspace, WorkspaceInfo

logger = logging.getLogger(__name__)


class CodeIngestPipeline:
    """Orchestrates code ingestion: chunking, embedding, and storage."""

    def __init__(
        self,
        storage: Storage,
        ollama_client: OllamaClient,
        config: AppConfig,
    ):
        """Initialize the ingestion pipeline.

        Args:
            storage: Storage instance for ChromaDB and SQLite.
            ollama_client: OllamaClient for generating embeddings.
            config: Application configuration.
        """
        self.storage = storage
        self.ollama_client = ollama_client
        self.config = config
        self.llm_tagger = LLMTagger(config.tagging, ollama_client, config.ollama)

    def ingest_file(
        self,
        file_path: str,
        workspace_id: str,
        force_reindex: bool = False,
    ) -> dict:
        """
        Ingest a single file: hash check, chunk, embed, store, and record dependencies.

        Process:
        1. Compute SHA-256 hash of file
        2. Check hash_cache: if identical and not force_reindex, skip
        3. If different (or not in cache), delete old chunks
        4. Chunk the file into CodeChunk objects
        5. Embed chunks in batch via Ollama
        6. Store chunks in ChromaDB and SQLite
        7. Extract and store dependencies
        8. Update hash_cache

        Args:
            file_path: Absolute path to the file to ingest.
            workspace_id: Workspace identifier.
            force_reindex: If True, ignore hash cache and re-index.

        Returns:
            Dictionary with keys:
                - indexed: bool (True if file was processed)
                - skipped: bool (True if file was skipped due to hash match)
                - chunks_created: int (number of chunks created)
                - errors: list[str] (any errors encountered)
        """
        result = {
            "indexed": False,
            "skipped": False,
            "chunks_created": 0,
            "errors": [],
        }

        file_path_abs = str(Path(file_path).resolve())

        try:
            # Check file size limit
            file_size_mb = Path(file_path_abs).stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.security.max_file_size_mb:
                result["skipped"] = True
                return result

            # Compute SHA-256 hash
            current_hash = hash_file(file_path_abs)

            # Check cache
            cached_hash = self.storage.get_file_hash(file_path_abs)
            if cached_hash == current_hash and not force_reindex:
                result["skipped"] = True
                return result

            # Delete old chunks if re-indexing
            self.storage.delete_file_chunks(file_path_abs, workspace_id)

            # Chunk the file
            try:
                chunks = chunk_file(file_path_abs, self.config)
            except Exception as e:
                result["errors"].append(f"Failed to chunk file: {str(e)}")
                return result

            if not chunks:
                # File type not supported or empty
                result["skipped"] = True
                return result

            # Apply heuristic H1 tagging to each chunk
            lang = detect_language(file_path_abs)
            for chunk in chunks:
                h1_tags = tag_h1(file_path_abs, lang, chunk.code)
                chunk.metadata["tags"] = h1_tags

            # Apply LLM H2 tagging once per file (add to all chunks)
            if self.config.tagging.h2_enabled:
                try:
                    source = Path(file_path_abs).read_text(encoding="utf-8", errors="ignore")
                    source_preview = "\n".join(source.split("\n")[:50])

                    if lang == "python":
                        imports = extract_imports_python(source)
                    elif lang:
                        imports = extract_imports_generic(source, lang)
                    else:
                        imports = []

                    symbols = [chunk.name for chunk in chunks]
                    h2_tags = self.llm_tagger.tag_h2(file_path_abs, source_preview, imports, symbols)

                    # Add H2 tags to all chunks
                    for chunk in chunks:
                        if "tags" not in chunk.metadata:
                            chunk.metadata["tags"] = []
                        chunk.metadata["tags"].extend(h2_tags)
                        chunk.metadata["tags"] = list(set(chunk.metadata["tags"]))  # Remove duplicates
                except Exception as e:
                    logger.warning(f"Failed to apply H2 tagging for {file_path_abs}: {e}")

            # Prepare batch embedding
            embed_texts = [chunk.to_embed_text(self.config.rag.max_embed_text_length) for chunk in chunks]

            # Embed batch
            try:
                embeddings = self.ollama_client.embed(embed_texts)
            except Exception as e:
                result["errors"].append(f"Failed to embed chunks: {str(e)}")
                return result

            if len(embeddings) != len(chunks):
                result["errors"].append(f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}")
                return result

            # Warn about zero-vector embeddings (chunks that failed individually)
            zero_count = sum(1 for emb in embeddings if all(v == 0.0 for v in emb))
            if zero_count > 0:
                logger.warning(
                    f"{zero_count}/{len(chunks)} chunks in {file_path_abs} "
                    "got zero-vector embeddings (still indexed for text search)"
                )

            # Store chunks
            for chunk, embedding in zip(chunks, embeddings):
                self.storage.store_chunk(chunk, embedding, workspace_id)

            # Extract and store dependencies
            try:
                lang = detect_language(file_path_abs)
                source = Path(file_path_abs).read_text(encoding="utf-8", errors="ignore")

                if lang == "python":
                    imports = extract_imports_python(source)
                elif lang:
                    imports = extract_imports_generic(source, lang)
                else:
                    imports = []

                for imp in imports:
                    # Store dependency: file -> import
                    self.storage.store_dependency(
                        source=file_path_abs,
                        target=imp,
                        workspace_id=workspace_id,
                    )
            except Exception as e:
                logger.warning(f"Failed to extract dependencies for {file_path_abs}: {e}")

            # Update hash cache
            self.storage.set_file_hash(file_path_abs, current_hash)

            result["indexed"] = True
            result["chunks_created"] = len(chunks)

        except Exception as e:
            result["errors"].append(f"Unexpected error: {str(e)}")

        return result

    def ingest_directory(
        self,
        dir_path: str,
        workspace: Optional[WorkspaceInfo] = None,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
        force_reindex: bool = False,
    ) -> dict:
        """
        Ingest a directory of source files.

        Process:
        1. Auto-detect workspace if not provided
        2. Walk directory tree with filters
        3. For each file, call ingest_file
        4. Track orphaned files and sync deletions (remove chunks for deleted files)
        5. Collect statistics by language
        6. Record scan event in scan_history

        Args:
            dir_path: Directory path (absolute or relative).
            workspace: WorkspaceInfo object. If None, auto-detects.
            recursive: If True, walk subdirectories.
            exclude_patterns: List of fnmatch patterns to exclude (uses config defaults if None).
            force_reindex: If True, re-index all files regardless of hash cache.

        Returns:
            Dictionary with keys:
                - files_scanned: int (total files examined)
                - files_indexed: int (files successfully indexed)
                - files_skipped: int (files skipped due to hash match or unsupported)
                - languages: dict[str, int] (count by language)
                - total_chunks: int (total chunks created)
                - duration_ms: int (total scan duration)
                - errors: list[str] (any errors encountered)
        """
        start_time = time.time()
        result = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "languages": defaultdict(int),
            "total_chunks": 0,
            "duration_ms": 0,
            "errors": [],
        }

        dir_path_abs = str(Path(dir_path).resolve())

        # Auto-detect workspace if not provided
        if workspace is None:
            workspace = detect_workspace(dir_path_abs)
            if workspace is None:
                result["errors"].append(f"Could not detect workspace for {dir_path_abs}")
                return result

        workspace_id = workspace.id

        # Use provided exclude patterns or config defaults
        if exclude_patterns is None:
            exclude_patterns = self.config.rag.default_exclude_patterns

        # Track processed files for orphan detection
        processed_files = set()

        # Walk directory
        try:
            if recursive:
                walk_iter = Path(dir_path_abs).rglob("*")
            else:
                walk_iter = Path(dir_path_abs).glob("*")

            for path in walk_iter:
                # Skip directories
                if path.is_dir():
                    continue

                result["files_scanned"] += 1

                # Check exclusion patterns
                if is_excluded(str(path), exclude_patterns):
                    result["files_skipped"] += 1
                    continue

                # Check file extension
                if path.suffix not in self.config.rag.supported_extensions:
                    result["files_skipped"] += 1
                    continue

                # Ingest file
                file_result = self.ingest_file(str(path), workspace_id, force_reindex)

                processed_files.add(str(path))

                if file_result["indexed"]:
                    result["files_indexed"] += 1
                    result["total_chunks"] += file_result["chunks_created"]

                    # Track language
                    lang = detect_language(str(path))
                    if lang:
                        result["languages"][lang] += 1

                if file_result["skipped"]:
                    result["files_skipped"] += 1

                if file_result["errors"]:
                    result["errors"].extend(file_result["errors"])

        except Exception as e:
            result["errors"].append(f"Failed to walk directory: {str(e)}")

        # Sync deletions: remove chunks for orphaned files
        if self.config.watcher.sync_deletions:
            try:
                self._sync_deletions(workspace_id, dir_path_abs, processed_files)
            except Exception as e:
                logger.warning(f"Failed to sync deletions: {e}")

        # Record scan event
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            self.storage.record_scan(
                workspace_id=workspace_id,
                files_scanned=result["files_scanned"],
                chunks_created=result["total_chunks"],
                duration_ms=duration_ms,
            )
            result["duration_ms"] = duration_ms
        except Exception as e:
            logger.warning(f"Failed to record scan: {e}")

        # Convert defaultdict to regular dict
        result["languages"] = dict(result["languages"])

        return result

    def _sync_deletions(
        self,
        workspace_id: str,
        dir_path: str,
        processed_files: set[str],
    ) -> None:
        """
        Remove chunks for files that were indexed but no longer exist in the directory.

        Args:
            workspace_id: Workspace identifier.
            dir_path: Directory path.
            processed_files: Set of file paths currently in the directory.
        """
        import sqlite3

        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()

        # Get all files indexed in this workspace
        cursor.execute(
            "SELECT DISTINCT file_path FROM path_index WHERE workspace_id = ?",
            (workspace_id,),
        )
        indexed_files = {row[0] for row in cursor.fetchall()}
        conn.close()

        # Find orphaned files (indexed but not in directory)
        dir_path_abs = str(Path(dir_path).resolve())
        orphaned = {
            f for f in indexed_files
            if f.startswith(dir_path_abs) and f not in processed_files
        }

        # Delete chunks for orphaned files
        for file_path in orphaned:
            try:
                self.storage.delete_file_chunks(file_path, workspace_id)
                logger.info(f"Deleted chunks for orphaned file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks for {file_path}: {e}")
