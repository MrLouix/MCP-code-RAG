"""Tests for CodeChunk dataclass and text file chunking."""

import pytest
from mcp_code_rag.chunker import CodeChunk, chunk_text_file


def test_codechunk_creation():
    """Test creating a basic CodeChunk."""
    chunk = CodeChunk(
        type="function",
        name="example",
        package="test_pkg",
        language="python",
        file_path="/path/to/file.py",
        start_line=1,
        end_line=10,
        selection_start=1,
        selection_end=1,
        signature="def example():",
        docstring="Test function.",
        code="def example():\n    pass",
    )

    assert chunk.type == "function"
    assert chunk.name == "example"
    assert chunk.package == "test_pkg"
    assert chunk.language == "python"


def test_to_embed_text_combines_parts():
    """Test that to_embed_text combines signature, docstring, and code."""
    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="Test function.",
        code="def test():\n    return 42",
    )

    embed_text = chunk.to_embed_text()

    assert "def test():" in embed_text
    assert "Test function." in embed_text
    assert "return 42" in embed_text


def test_to_embed_text_truncation():
    """Test that to_embed_text truncates to max_length."""
    long_code = "x = 1\n" * 2000

    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=1000,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="Test.",
        code=long_code,
    )

    embed_text = chunk.to_embed_text(max_length=500)

    assert len(embed_text) <= 500


def test_to_embed_text_custom_max_length():
    """Test to_embed_text with custom max_length."""
    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="Test.",
        code="x = 1\ny = 2",
    )

    embed_text = chunk.to_embed_text(max_length=20)

    assert len(embed_text) == 20


def test_to_embed_text_without_docstring():
    """Test to_embed_text when docstring is empty."""
    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="",
        code="return 42",
    )

    embed_text = chunk.to_embed_text()

    assert "def test():" in embed_text
    assert "return 42" in embed_text


def test_to_embed_text_without_code():
    """Test to_embed_text when code is empty."""
    chunk = CodeChunk(
        type="class",
        name="MyClass",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="class MyClass:",
        docstring="A class.",
        code="",
    )

    embed_text = chunk.to_embed_text()

    assert "class MyClass:" in embed_text
    assert "A class." in embed_text


def test_to_chromadb_metadata_basic():
    """Test that to_chromadb_metadata returns JSON-safe dict."""
    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="Test.",
        code="return 42",
    )

    metadata = chunk.to_chromadb_metadata()

    assert isinstance(metadata, dict)
    assert metadata["type"] == "function"
    assert metadata["name"] == "test"
    assert metadata["package"] == "pkg"
    assert metadata["language"] == "python"
    assert metadata["file_path"] == "/test.py"
    assert metadata["start_line"] == 1
    assert metadata["end_line"] == 5


def test_to_chromadb_metadata_with_extra_metadata():
    """Test that extra metadata is serialized correctly."""
    chunk = CodeChunk(
        type="function",
        name="test",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=5,
        selection_start=1,
        selection_end=1,
        signature="def test():",
        docstring="Test.",
        code="return 42",
        metadata={
            "is_async": True,
            "params": ["x", "y"],
            "tags": ["important", "deprecated"],
        },
    )

    metadata = chunk.to_chromadb_metadata()

    # Simple values should be included directly
    assert metadata["meta_is_async"] is True

    # Complex values should be JSON-serialized
    assert "meta_params" in metadata
    assert "meta_tags" in metadata


def test_chunk_text_file_markdown():
    """Test chunking a markdown file."""
    source = """# README

This is a markdown file.

## Section 1

Some content here.
"""

    chunks = chunk_text_file(source, "/test/README.md", "doc")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "doc"
    assert chunk.name == "README.md"
    assert chunk.language == "doc"
    assert "# README" in chunk.code


def test_chunk_text_file_config_yaml():
    """Test chunking a YAML config file."""
    source = """
ollama:
  base_url: http://localhost:11434
  embed_model: nomic-embed-text
"""

    chunks = chunk_text_file(source, "/test/config.yaml", "config")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "config"
    assert chunk.name == "config.yaml"
    assert chunk.language == "config"


def test_chunk_text_file_json():
    """Test chunking a JSON file."""
    source = """{
  "name": "test-project",
  "version": "1.0.0"
}"""

    chunks = chunk_text_file(source, "/test/package.json", "config")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.type == "config"
    assert chunk.name == "package.json"


def test_chunk_text_file_sets_correct_line_count():
    """Test that line counts are set correctly."""
    source = "line1\nline2\nline3\nline4\nline5"

    chunks = chunk_text_file(source, "/test/file.txt", "doc")

    chunk = chunks[0]
    assert chunk.start_line == 1
    assert chunk.end_line == 5


def test_chunk_text_file_empty():
    """Test chunking an empty text file."""
    source = ""

    chunks = chunk_text_file(source, "/test/empty.txt", "doc")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.code == ""
    assert chunk.end_line == 1  # Empty file is 1 line


def test_codechunk_metadata_serialization():
    """Test that complex metadata is properly serialized."""
    chunk = CodeChunk(
        type="class",
        name="TestClass",
        package="pkg",
        language="python",
        file_path="/test.py",
        start_line=1,
        end_line=10,
        selection_start=1,
        selection_end=1,
        signature="class TestClass:",
        metadata={
            "methods": ["method1", "method2"],
            "is_abstract": False,
            "base_classes": {"Base1": "module1", "Base2": "module2"},
        },
    )

    metadata = chunk.to_chromadb_metadata()

    # All values should be serializable
    assert isinstance(metadata["meta_is_abstract"], bool)
    # Complex types should be JSON strings
    assert isinstance(metadata.get("meta_methods", ""), str)
    assert isinstance(metadata.get("meta_base_classes", ""), str)


def test_chunk_text_file_long_content():
    """Test chunking a file with many lines."""
    lines = ["line " + str(i) for i in range(1000)]
    source = "\n".join(lines)

    chunks = chunk_text_file(source, "/test/longfile.txt", "doc")

    chunk = chunks[0]
    assert chunk.end_line == 1000
    assert len(chunk.code.split("\n")) == 1000
