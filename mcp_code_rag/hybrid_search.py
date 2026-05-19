"""Hybrid search combining BM25 (keyword) and vector (semantic) search."""

import json
import logging
import sqlite3
from typing import Optional

from mcp_code_rag.config import HybridSearchConfig
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.storage import Storage

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid search combining BM25 (full-text) and vector (semantic) search."""

    def __init__(
        self,
        storage: Storage,
        ollama_client: OllamaClient,
        config: Optional[HybridSearchConfig] = None,
    ):
        """Initialize HybridSearch with storage, embeddings client, and config.

        Args:
            storage: Storage instance for accessing ChromaDB and SQLite.
            ollama_client: OllamaClient for generating embeddings.
            config: HybridSearchConfig instance (uses defaults if None).
        """
        self.storage = storage
        self.ollama_client = ollama_client
        self.config = config or HybridSearchConfig()

    def bm25_search(
        self,
        query: str,
        workspace_id: str,
        language_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Search using BM25 (full-text search) via FTS5.

        Args:
            query: Search query string.
            workspace_id: Workspace identifier.
            language_filter: Optional language filter (e.g., "python").
            top_k: Maximum number of results.

        Returns:
            List of search results with normalized BM25 scores (0-1).
        """
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Escape query for FTS5
        fts_query = f'"{query}"'

        # Base FTS5 query
        sql = f"""
            SELECT rowid, workspace_id, file_path, symbol_name, code, docstring, rank
            FROM {self.config.fts5_table}
            WHERE {self.config.fts5_table} MATCH ?
            AND workspace_id = ?
        """
        params = [fts_query, workspace_id]

        # Add language filter if provided
        if language_filter:
            sql += " AND file_path LIKE ?"
            # Language detection via file extension mapping
            lang_extensions = {
                "python": "%.py",
                "javascript": "%.js",
                "typescript": "%.ts",
                "go": "%.go",
                "java": "%.java",
                "rust": "%.rs",
                "csharp": "%.cs",
                "cpp": "%.cpp",
                "c": "%.c",
                "ruby": "%.rb",
                "php": "%.php",
            }
            extension = lang_extensions.get(language_filter.lower(), f"%.{language_filter}")
            params.append(extension)

        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        finally:
            conn.close()

        # Normalize BM25 scores (rank is negative, lower is better)
        results = []
        if rows:
            # Find min rank (most negative)
            min_rank = min(row["rank"] for row in rows)
            max_rank = max(row["rank"] for row in rows)
            rank_range = max_rank - min_rank if max_rank != min_rank else 1

            for row in rows:
                # Normalize rank to [0, 1] (higher is better for score)
                normalized_score = 1.0 - (row["rank"] - min_rank) / rank_range
                results.append(
                    {
                        "file_path": row["file_path"],
                        "symbol_name": row["symbol_name"],
                        "code": row["code"],
                        "docstring": row["docstring"],
                        "bm25_score": normalized_score,
                        "workspace_id": row["workspace_id"],
                    }
                )

        return results

    def vector_search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        language_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Search using vector similarity via ChromaDB.

        Args:
            query_embedding: Query embedding vector.
            workspace_id: Workspace identifier.
            language_filter: Optional language filter (e.g., "python").
            top_k: Maximum number of results.

        Returns:
            List of search results with normalized distance scores (0-1).
        """
        results = []

        # Get all collections for this workspace
        try:
            collections = self.storage.chroma_client.list_collections()
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return results

        # Filter collections by workspace and language
        for collection_info in collections:
            coll_name = collection_info.name
            if not coll_name.startswith(f"{workspace_id}_"):
                continue

            # Extract language from collection name: {workspace_id}_{lang}
            lang = coll_name[len(workspace_id) + 1 :]

            if language_filter and lang.lower() != language_filter.lower():
                continue

            try:
                collection = self.storage.chroma_client.get_collection(coll_name)
                # Request more results to account for merging from multiple collections
                query_results = collection.query(
                    query_embeddings=[query_embedding], n_results=top_k * 2, include=["distances", "metadatas", "documents"]
                )

                if query_results["distances"] and query_results["distances"][0]:
                    for distance, metadata, document in zip(
                        query_results["distances"][0],
                        query_results["metadatas"][0],
                        query_results["documents"][0],
                    ):
                        # Normalize distance to [0, 1] (1 - distance for similarity score)
                        normalized_distance = 1.0 - distance
                        results.append(
                            {
                                "file_path": metadata.get("file_path", ""),
                                "symbol_name": metadata.get("name", ""),
                                "code": document,
                                "docstring": metadata.get("docstring", ""),
                                "distance": distance,
                                "vector_score": normalized_distance,
                                "workspace_id": workspace_id,
                                "metadata": metadata,
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to query collection {coll_name}: {e}")

        # Sort by distance and return top_k
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        workspace_id: str = "default",
        language_filter: Optional[str] = None,
        file_filter: Optional[str] = None,
        exclude_tests: bool = False,
        tag_filter: Optional[list[str]] = None,
        tags_mode: str = "any",
    ) -> list[dict]:
        """Hybrid search combining BM25 and vector search.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            workspace_id: Workspace identifier.
            language_filter: Optional language filter (e.g., "python").
            file_filter: Optional file path filter (substring match).
            exclude_tests: If True, exclude files containing "test" in path.
            tag_filter: Optional list of tags to filter by.
            tags_mode: "any" (OR) or "all" (AND) for tag filtering.

        Returns:
            List of hybrid search results with combined scores, sorted by score.
        """
        # Generate embedding for query
        try:
            embeddings = self.ollama_client.embed([query])
            if not embeddings:
                logger.warning("Failed to generate embedding for query")
                return []
            query_embedding = embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Run both searches
        bm25_results = self.bm25_search(query, workspace_id, language_filter, top_k * 2)
        vector_results = self.vector_search(query_embedding, workspace_id, language_filter, top_k * 2)

        # Merge results by file_path + symbol_name
        merged = {}
        for result in bm25_results:
            key = (result["file_path"], result["symbol_name"])
            if key not in merged:
                merged[key] = {
                    "file_path": result["file_path"],
                    "symbol_name": result["symbol_name"],
                    "code": result["code"],
                    "docstring": result["docstring"],
                    "workspace_id": result["workspace_id"],
                    "bm25_score": result["bm25_score"],
                    "vector_score": 0.0,
                    "metadata": {},
                }
            else:
                merged[key]["bm25_score"] = result["bm25_score"]

        for result in vector_results:
            key = (result["file_path"], result["symbol_name"])
            if key not in merged:
                merged[key] = {
                    "file_path": result["file_path"],
                    "symbol_name": result["symbol_name"],
                    "code": result["code"],
                    "docstring": result["docstring"],
                    "workspace_id": result["workspace_id"],
                    "bm25_score": 0.0,
                    "vector_score": result["vector_score"],
                    "metadata": result.get("metadata", {}),
                }
            else:
                merged[key]["vector_score"] = result["vector_score"]
                merged[key]["metadata"] = result.get("metadata", {})

        # Calculate hybrid score and apply filters
        final_results = []
        for key, result in merged.items():
            # Skip if filters don't match
            if file_filter and file_filter not in result["file_path"]:
                continue

            if exclude_tests and "test" in result["file_path"].lower():
                continue

            # Tag filtering
            if tag_filter:
                # Extract tags from metadata
                meta_tags = result["metadata"].get("meta_tags")
                if meta_tags:
                    if isinstance(meta_tags, str):
                        try:
                            meta_tags = json.loads(meta_tags)
                        except (json.JSONDecodeError, TypeError):
                            meta_tags = []
                else:
                    meta_tags = []

                if tags_mode == "all":
                    # All filter tags must be present
                    if not all(tag in meta_tags for tag in tag_filter):
                        continue
                else:  # "any"
                    # At least one filter tag must be present
                    if not any(tag in meta_tags for tag in tag_filter):
                        continue

            # Calculate hybrid score
            hybrid_score = (
                self.config.alpha * result["vector_score"] + self.config.beta * result["bm25_score"]
            )
            result["hybrid_score"] = hybrid_score

            final_results.append(result)

        # Sort by hybrid score and return top_k
        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return final_results[:top_k]
