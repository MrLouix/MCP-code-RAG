"""Tests for LLM-based tagging (H2)."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_code_rag.config import TaggingConfig, TaxonomyConfig, OllamaConfig
from mcp_code_rag.ollama_client import OllamaClient
from mcp_code_rag.tagging.llm_tagger import LLMTagger


@pytest.fixture
def temp_cache_db():
    """Create a temporary cache database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def tagging_config(temp_cache_db):
    """Create a TaggingConfig for testing."""
    return TaggingConfig(
        h2_enabled=True,
        use_cache=True,
        cache_path=temp_cache_db,
        taxonomy=TaxonomyConfig(),
    )


@pytest.fixture
def ollama_config():
    """Create an OllamaConfig for testing."""
    return OllamaConfig(tag_model="qwen3.5")


@pytest.fixture
def ollama_client():
    """Create a mock OllamaClient."""
    client = MagicMock(spec=OllamaClient)
    return client


def test_llm_tagger_disabled(tagging_config, ollama_client, ollama_config):
    """Test that h2_enabled=False returns empty list without network calls."""
    tagging_config.h2_enabled = False
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    result = tagger.tag_h2(
        file_path="test.py",
        source_preview="print('hello')",
        imports=[],
        symbols=["print"],
    )

    assert result == []
    ollama_client.generate.assert_not_called()


def test_llm_tagger_valid_response(tagging_config, ollama_client, ollama_config):
    """Test parsing and validating LLM response."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response = json.dumps({
        "tags": [
            "framework:fastapi",
            "design_pattern:repository",
            "purpose:api",
        ]
    })
    ollama_client.generate.return_value = response

    result = tagger.tag_h2(
        file_path="api/main.py",
        source_preview="from fastapi import FastAPI\napp = FastAPI()",
        imports=["fastapi"],
        symbols=["app"],
    )

    assert "framework:fastapi" in result
    assert "design_pattern:repository" in result
    assert "purpose:api" in result
    ollama_client.generate.assert_called_once()


def test_llm_tagger_rejects_invalid_tags(tagging_config, ollama_client, ollama_config):
    """Test that tags outside taxonomy are rejected."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response = json.dumps({
        "tags": [
            "framework:fastapi",
            "framework:unknown_framework",  # Invalid
            "design_pattern:repository",
            "design_pattern:invalid_pattern",  # Invalid
        ]
    })
    ollama_client.generate.return_value = response

    result = tagger.tag_h2(
        file_path="api/main.py",
        source_preview="from fastapi import FastAPI",
        imports=["fastapi"],
        symbols=["app"],
    )

    assert "framework:fastapi" in result
    assert "design_pattern:repository" in result
    assert "framework:unknown_framework" not in result
    assert "design_pattern:invalid_pattern" not in result


def test_llm_tagger_cache_round_trip(tagging_config, ollama_client, ollama_config):
    """Test cache write and read."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response = json.dumps({
        "tags": ["framework:django", "purpose:util"]
    })
    ollama_client.generate.return_value = response

    # First call - should hit LLM
    result1 = tagger.tag_h2(
        file_path="utils/helpers.py",
        source_preview="def helper():\n    pass",
        imports=[],
        symbols=["helper"],
    )

    assert "framework:django" in result1
    assert ollama_client.generate.call_count == 1

    # Second call with same file - should hit cache
    result2 = tagger.tag_h2(
        file_path="utils/helpers.py",
        source_preview="def helper():\n    pass",
        imports=[],
        symbols=["helper"],
    )

    assert result1 == result2
    assert ollama_client.generate.call_count == 1  # No additional call


def test_llm_tagger_cache_miss_different_content(tagging_config, ollama_client, ollama_config):
    """Test cache miss when file content changes."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response1 = json.dumps({"tags": ["framework:fastapi"]})
    response2 = json.dumps({"tags": ["framework:django"]})
    ollama_client.generate.side_effect = [response1, response2]

    # First call
    result1 = tagger.tag_h2(
        file_path="app.py",
        source_preview="from fastapi import FastAPI",
        imports=["fastapi"],
        symbols=["app"],
    )
    assert "framework:fastapi" in result1

    # Second call with different content - cache miss
    result2 = tagger.tag_h2(
        file_path="app.py",
        source_preview="from django import forms",
        imports=["django"],
        symbols=["form"],
    )
    assert "framework:django" in result2
    assert ollama_client.generate.call_count == 2


def test_llm_tagger_invalid_json_response(tagging_config, ollama_client, ollama_config):
    """Test error handling for invalid JSON."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    ollama_client.generate.return_value = "not valid json"

    with pytest.raises(ValueError, match="Invalid JSON"):
        tagger.tag_h2(
            file_path="test.py",
            source_preview="print()",
            imports=[],
            symbols=["print"],
        )


def test_llm_tagger_missing_tags_key(tagging_config, ollama_client, ollama_config):
    """Test that missing tags key results in empty list."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    ollama_client.generate.return_value = json.dumps({"result": []})

    result = tagger.tag_h2(
        file_path="test.py",
        source_preview="print()",
        imports=[],
        symbols=["print"],
    )

    assert result == []


def test_compute_file_hash_consistency(tagging_config, ollama_client, ollama_config):
    """Test that file hash is consistent."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    hash1 = tagger._compute_file_hash("test.py", "content")
    hash2 = tagger._compute_file_hash("test.py", "content")
    hash3 = tagger._compute_file_hash("test.py", "different")

    assert hash1 == hash2
    assert hash1 != hash3


def test_validate_tags_allows_common_categories(tagging_config, ollama_client, ollama_config):
    """Test that common tag categories are allowed."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    tags = [
        "lang:python",
        "type:test",
        "layer:api",
        "framework:fastapi",
    ]

    validated = tagger._validate_tags(tags)

    assert "lang:python" in validated
    assert "type:test" in validated
    assert "layer:api" in validated
    assert "framework:fastapi" in validated


def test_cache_disabled_no_database_access(tagging_config, ollama_client, ollama_config):
    """Test that cache is skipped when disabled."""
    tagging_config.use_cache = False
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response = json.dumps({"tags": ["framework:flask"]})
    ollama_client.generate.return_value = response

    result = tagger.tag_h2(
        file_path="app.py",
        source_preview="from flask import Flask",
        imports=["flask"],
        symbols=["app"],
    )

    assert "framework:flask" in result


def test_build_prompt_structure(tagging_config, ollama_client, ollama_config):
    """Test that prompt contains expected elements."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    imports = ["fastapi", "sqlalchemy"]
    symbols = ["app", "get_user", "User"]
    preview = "from fastapi import FastAPI\napp = FastAPI()"

    prompt = tagger._build_prompt(
        file_path="main.py",
        source_preview=preview,
        imports=imports,
        symbols=symbols,
    )

    assert "FILE: main.py" in prompt
    assert "fastapi" in prompt
    assert "sqlalchemy" in prompt
    assert "app" in prompt
    assert "get_user" in prompt
    assert "User" in prompt
    assert "framework" in prompt
    assert "design_pattern" in prompt


def test_parse_response_whitespace_handling(tagging_config, ollama_client, ollama_config):
    """Test that JSON with surrounding whitespace is parsed."""
    tagger = LLMTagger(tagging_config, ollama_client, ollama_config)

    response = '  \n{"tags": ["framework:fastapi"]}\n  '

    tags = tagger._parse_response(response)

    assert tags == ["framework:fastapi"]


def test_cache_persistence(temp_cache_db):
    """Test that cache persists across tagger instances."""
    config = TaggingConfig(
        h2_enabled=True,
        use_cache=True,
        cache_path=temp_cache_db,
    )
    ollama_config = OllamaConfig()
    mock_client = MagicMock(spec=OllamaClient)

    # First tagger instance - write to cache
    tagger1 = LLMTagger(config, mock_client, ollama_config)
    file_hash = tagger1._compute_file_hash("test.py", "content")
    tagger1.cache_tags(file_hash, ["framework:fastapi", "purpose:api"])

    # Second tagger instance - read from cache
    tagger2 = LLMTagger(config, mock_client, ollama_config)
    cached = tagger2.get_cached_tags(file_hash)

    assert cached == ["framework:fastapi", "purpose:api"]
