"""Tests for hybrid search (BM25 + vector search)."""

import json
from unittest.mock import Mock, patch

import pytest

from mcp_code_rag.chunker import CodeChunk
from mcp_code_rag.config import HybridSearchConfig, OllamaConfig
from mcp_code_rag.hybrid_search import HybridSearch
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.storage import Storage


@pytest.fixture
def storage(tmp_path):
    """Create a Storage instance with temp directory."""
    return Storage(str(tmp_path))


@pytest.fixture
def ollama_config():
    """Create a test OllamaConfig."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        timeout_s=5.0,
        embed_timeout_s=10.0,
        max_retries=2,
    )


@pytest.fixture
def ollama_client(ollama_config):
    """Create an OllamaClient instance for testing."""
    return OllamaClient(ollama_config)


@pytest.fixture
def hybrid_search_config():
    """Create a HybridSearchConfig."""
    return HybridSearchConfig(alpha=0.6, beta=0.4)


@pytest.fixture
def hybrid_search(storage, ollama_client, hybrid_search_config):
    """Create a HybridSearch instance."""
    return HybridSearch(storage, ollama_client, hybrid_search_config)


@pytest.fixture
def sample_chunks():
    """Create sample CodeChunk objects for testing."""
    return [
        CodeChunk(
            type="function",
            name="process_data",
            package="data_processor",
            language="python",
            file_path="/workspace/data_processor.py",
            start_line=10,
            end_line=25,
            selection_start=10,
            selection_end=10,
            signature="def process_data(data: list) -> dict:",
            docstring="Process input data and return results.",
            code="def process_data(data: list) -> dict:\n    return {'processed': data}",
            metadata={"visibility": "public", "tags": ["data", "processing"]},
        ),
        CodeChunk(
            type="function",
            name="fetch_user",
            package="api",
            language="python",
            file_path="/workspace/api/user_api.py",
            start_line=5,
            end_line=12,
            selection_start=5,
            selection_end=5,
            signature="def fetch_user(user_id: int) -> dict:",
            docstring="Fetch user information from database.",
            code="def fetch_user(user_id: int) -> dict:\n    return db.get_user(user_id)",
            metadata={"visibility": "public", "tags": ["user", "api"]},
        ),
        CodeChunk(
            type="function",
            name="test_process",
            package="tests",
            language="python",
            file_path="/workspace/tests/test_processor.py",
            start_line=1,
            end_line=10,
            selection_start=1,
            selection_end=1,
            signature="def test_process():",
            docstring="Test process_data function.",
            code="def test_process():\n    assert process_data([1,2,3]) == {'processed': [1,2,3]}",
            metadata={"visibility": "public", "tags": ["test"]},
        ),
        CodeChunk(
            type="function",
            name="handle_request",
            package="handler",
            language="javascript",
            file_path="/workspace/handler/request.js",
            start_line=20,
            end_line=35,
            selection_start=20,
            selection_end=20,
            signature="function handle_request(req) {",
            docstring="Handle HTTP request.",
            code="function handle_request(req) {\n  return processRequest(req);\n}",
            metadata={"visibility": "public", "tags": ["handler"]},
        ),
    ]


class TestBM25Search:
    """Tests for BM25 (full-text) search."""

    def test_bm25_exact_match_scores_high(self, storage, hybrid_search, sample_chunks):
        """Verify exact match in BM25 scores highest."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Search for exact symbol name
        results = hybrid_search.bm25_search("process_data", workspace_id, top_k=10)

        assert len(results) > 0
        # First result should be the exact match
        assert results[0]["symbol_name"] == "process_data"
        assert results[0]["bm25_score"] > 0.0

    def test_bm25_language_filter(self, storage, hybrid_search, sample_chunks):
        """Verify language_filter excludes other languages."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Search with language filter
        results = hybrid_search.bm25_search("function", workspace_id, language_filter="python", top_k=10)

        # All results should be Python files
        for result in results:
            assert result["file_path"].endswith(".py")

    def test_bm25_empty_results(self, storage, hybrid_search):
        """Verify empty results for non-existent query."""
        workspace_id = "ws1"
        results = hybrid_search.bm25_search("nonexistent_xyz_query", workspace_id, top_k=10)
        assert results == []

    def test_bm25_respects_top_k(self, storage, hybrid_search, sample_chunks):
        """Verify top_k limit is respected."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Search with top_k=2
        results = hybrid_search.bm25_search("function", workspace_id, top_k=2)
        assert len(results) <= 2


class TestVectorSearch:
    """Tests for vector (semantic) search."""

    def test_vector_search_returns_results(self, storage, hybrid_search, sample_chunks):
        """Verify vector search returns results."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Query with same embedding should return results
        results = hybrid_search.vector_search(embedding, workspace_id, top_k=10)

        assert len(results) > 0
        for result in results:
            assert "file_path" in result
            assert "symbol_name" in result
            assert "vector_score" in result

    def test_vector_search_language_filter(self, storage, hybrid_search, sample_chunks):
        """Verify language_filter excludes other languages."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Search with language filter
        results = hybrid_search.vector_search(embedding, workspace_id, language_filter="python", top_k=10)

        # All results should be from Python collections
        for result in results:
            assert "python" in result["file_path"].lower() or result["file_path"].endswith(".py")

    def test_vector_search_respects_top_k(self, storage, hybrid_search, sample_chunks):
        """Verify top_k limit is respected."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Search with top_k=2
        results = hybrid_search.vector_search(embedding, workspace_id, top_k=2)
        assert len(results) <= 2


class TestHybridSearch:
    """Tests for hybrid search combining BM25 and vector search."""

    def test_hybrid_search_respects_top_k(self, storage, hybrid_search, sample_chunks):
        """Verify hybrid_search respects top_k limit."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search("process data", top_k=2, workspace_id=workspace_id)

            assert len(results) <= 2
            for result in results:
                assert "hybrid_score" in result

    def test_hybrid_search_language_filter_python(self, storage, hybrid_search, sample_chunks):
        """Verify language_filter=python excludes other languages."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search(
                "function", top_k=10, workspace_id=workspace_id, language_filter="python"
            )

            # All results should be Python files
            for result in results:
                assert result["file_path"].endswith(".py")

    def test_hybrid_search_exclude_tests(self, storage, hybrid_search, sample_chunks):
        """Verify exclude_tests filters out test files."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search(
                "function", top_k=10, workspace_id=workspace_id, exclude_tests=True
            )

            # No test files should be in results
            for result in results:
                assert "test" not in result["file_path"].lower()

    def test_hybrid_search_file_filter(self, storage, hybrid_search, sample_chunks):
        """Verify file_filter filters by path substring."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search(
                "function", top_k=10, workspace_id=workspace_id, file_filter="api"
            )

            # All results should contain "api" in file_path
            for result in results:
                assert "api" in result["file_path"].lower()

    def test_hybrid_search_tags_mode_all(self, storage, hybrid_search, sample_chunks):
        """Verify tags_mode=all requires all tags to be present."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search(
                "process", top_k=10, workspace_id=workspace_id, tag_filter=["data", "processing"], tags_mode="all"
            )

            # Results should only include chunks with both "data" AND "processing" tags
            for result in results:
                meta_tags = result["metadata"].get("meta_tags")
                if isinstance(meta_tags, str):
                    meta_tags = json.loads(meta_tags)
                assert "data" in meta_tags and "processing" in meta_tags

    def test_hybrid_search_tags_mode_any(self, storage, hybrid_search, sample_chunks):
        """Verify tags_mode=any includes items with any of the tags."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search(
                "test user", top_k=10, workspace_id=workspace_id, tag_filter=["test", "user"], tags_mode="any"
            )

            # Results should include chunks with "test" OR "user" tags
            for result in results:
                meta_tags = result["metadata"].get("meta_tags")
                if isinstance(meta_tags, str):
                    meta_tags = json.loads(meta_tags)
                assert "test" in meta_tags or "user" in meta_tags

    def test_hybrid_search_scoring(self, storage, hybrid_search, sample_chunks):
        """Verify hybrid score calculation."""
        workspace_id = "ws1"
        embedding = [0.1] * 768

        # Store chunks
        for chunk in sample_chunks:
            storage.store_chunk(chunk, embedding, workspace_id)

        # Mock the embed method
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.return_value = [[0.2] * 768]

            results = hybrid_search.hybrid_search("data", top_k=10, workspace_id=workspace_id)

            # Verify hybrid scores are calculated
            for result in results:
                assert "hybrid_score" in result
                # Score should be: 0.6 * vector_score + 0.4 * bm25_score
                expected = 0.6 * result["vector_score"] + 0.4 * result["bm25_score"]
                assert abs(result["hybrid_score"] - expected) < 1e-6

    def test_hybrid_search_embedding_failure(self, storage, hybrid_search):
        """Verify graceful handling of embedding failures."""
        workspace_id = "ws1"

        # Mock the embed method to fail
        with patch.object(hybrid_search.ollama_client, "embed") as mock_embed:
            mock_embed.side_effect = Exception("Embedding service unavailable")

            results = hybrid_search.hybrid_search("test query", workspace_id=workspace_id)

            assert results == []
