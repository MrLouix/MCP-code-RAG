"""Tests for project configuration."""

import tomllib
import yaml
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from mcp_code_rag.config import (
    AppConfig,
    HybridSearchConfig,
    OllamaConfig,
    SecurityConfig,
    load_config,
)


def test_mcp_code_rag_imports():
    """Test that mcp_code_rag package can be imported."""
    import mcp_code_rag
    assert mcp_code_rag.__version__ == "0.1.0"


def test_pyproject_toml_parse():
    """Test that pyproject.toml is valid and contains required dependencies."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    assert config["project"]["name"] == "mcp-code-rag"
    assert config["project"]["version"] == "0.1.0"

    # Check required dependencies
    required_deps = {
        "fastmcp",
        "chromadb",
        "watchdog",
        "pydantic",
        "pydantic-settings",
        "httpx",
        "pyyaml",
    }

    dependencies = config["project"]["dependencies"]
    dep_names = {dep.split(">")[0].split("<")[0].split("=")[0] for dep in dependencies}

    assert required_deps.issubset(dep_names), f"Missing dependencies: {required_deps - dep_names}"

    # Check dev dependencies
    dev_deps = config["project"]["optional-dependencies"]["dev"]
    dev_dep_names = {dep.split(">")[0].split("<")[0].split("=")[0] for dep in dev_deps}

    required_dev_deps = {"pytest", "pytest-asyncio", "pytest-cov"}
    assert required_dev_deps.issubset(dev_dep_names)


def test_config_example_yaml_valid():
    """Test that config.example.yaml is valid YAML."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check main sections exist
    required_sections = {
        "ollama",
        "rag",
        "hybrid_search",
        "tagging",
        "watcher",
        "workspace",
        "security",
        "logging",
    }

    assert required_sections.issubset(config.keys()), \
        f"Missing sections: {required_sections - set(config.keys())}"

    # Check ollama config
    assert "base_url" in config["ollama"]
    assert "embed_model" in config["ollama"]
    assert "tag_model" in config["ollama"]

    # Check rag config
    assert "index_path" in config["rag"]
    assert "chunk_strategy" in config["rag"]
    assert "supported_extensions" in config["rag"]
    assert "default_exclude_patterns" in config["rag"]

    # Check hybrid_search config
    assert config["hybrid_search"]["enabled"] is True
    assert "alpha" in config["hybrid_search"]
    assert "beta" in config["hybrid_search"]

    # Check tagging config
    assert "auto_tag_enabled" in config["tagging"]
    assert "taxonomy" in config["tagging"]

    # Check watcher config
    assert "enabled" in config["watcher"]
    assert "debounce_ms" in config["watcher"]

    # Check workspace config
    assert "collection_prefix" in config["workspace"]

    # Check security config
    assert "max_file_size_mb" in config["security"]

    # Check logging config
    assert "level" in config["logging"]
    assert "format" in config["logging"]


# Tests for load_config and Pydantic models


def test_load_config_example_yaml():
    """Test loading config.example.yaml produces a valid AppConfig."""
    config_path = Path(__file__).parent.parent / "config.example.yaml"
    app_config = load_config(str(config_path))

    assert isinstance(app_config, AppConfig)
    assert isinstance(app_config.ollama, OllamaConfig)
    assert app_config.ollama.base_url == "http://172.28.128.1:11434"
    assert app_config.ollama.embed_model == "nomic-embed-text"

    assert app_config.rag.chunk_strategy == "ast"
    assert len(app_config.rag.supported_extensions) > 0

    assert app_config.hybrid_search.alpha == 0.6
    assert app_config.hybrid_search.beta == 0.4

    assert app_config.security.max_file_size_mb == 10.0


def test_load_config_minimal_yaml():
    """Test loading minimal YAML with only ollama section uses defaults for others."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"ollama": {"base_url": "http://custom:11434"}}, f)
        f.flush()

        try:
            app_config = load_config(f.name)

            assert isinstance(app_config, AppConfig)
            assert app_config.ollama.base_url == "http://custom:11434"
            # Other sections should use defaults (path is resolved to absolute)
            assert "code_rag_index" in app_config.rag.index_path
            assert app_config.tagging.auto_tag_enabled is True
            assert app_config.logging.level == "INFO"
        finally:
            Path(f.name).unlink()


def test_load_config_nonexistent_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_default_when_no_path():
    """Test that load_config without path loads defaults when config.yaml missing."""
    # When config.yaml doesn't exist and no path is given, should use defaults
    app_config = load_config(None)
    assert isinstance(app_config, AppConfig)
    assert app_config.ollama.base_url == "http://localhost:11434"


def test_hybrid_search_weights_sum_to_one():
    """Test that alpha + beta must equal 1.0."""
    # Valid case
    config = HybridSearchConfig(alpha=0.6, beta=0.4)
    assert config.alpha == 0.6
    assert config.beta == 0.4

    # Invalid case: alpha + beta != 1.0
    with pytest.raises(ValueError, match="alpha \\+ beta must equal 1.0"):
        HybridSearchConfig(alpha=0.6, beta=0.3)

    with pytest.raises(ValueError, match="alpha \\+ beta must equal 1.0"):
        HybridSearchConfig(alpha=0.7, beta=0.5)


def test_hybrid_search_weights_range():
    """Test that alpha and beta must be between 0 and 1."""
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        HybridSearchConfig(alpha=1.5, beta=-0.5)

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        HybridSearchConfig(alpha=-0.1, beta=1.1)


def test_security_config_positive_file_size():
    """Test that max_file_size_mb must be positive."""
    # Valid case
    config = SecurityConfig(max_file_size_mb=10.0)
    assert config.max_file_size_mb == 10.0

    # Invalid cases
    with pytest.raises(ValueError, match="must be positive"):
        SecurityConfig(max_file_size_mb=0)

    with pytest.raises(ValueError, match="must be positive"):
        SecurityConfig(max_file_size_mb=-5.0)


def test_load_config_preserves_weights():
    """Test that config loading preserves alpha + beta = 1.0 constraint."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "hybrid_search": {
                    "enabled": True,
                    "alpha": 0.7,
                    "beta": 0.3,
                }
            },
            f,
        )
        f.flush()

        try:
            app_config = load_config(f.name)
            assert app_config.hybrid_search.alpha == 0.7
            assert app_config.hybrid_search.beta == 0.3
        finally:
            Path(f.name).unlink()


def test_load_config_invalid_weights():
    """Test that loading config with invalid weights raises error."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "hybrid_search": {
                    "alpha": 0.6,
                    "beta": 0.3,
                }
            },
            f,
        )
        f.flush()

        try:
            with pytest.raises(ValueError, match="alpha \\+ beta must equal 1.0"):
                load_config(f.name)
        finally:
            Path(f.name).unlink()


def test_load_config_invalid_file_size():
    """Test that loading config with invalid file size raises error."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "security": {
                    "max_file_size_mb": -1,
                }
            },
            f,
        )
        f.flush()

        try:
            with pytest.raises(ValueError, match="must be positive"):
                load_config(f.name)
        finally:
            Path(f.name).unlink()


def test_ollama_config_positive_timeouts():
    """Test that Ollama timeouts must be positive."""
    # Valid case
    config = OllamaConfig(timeout_s=30.0, embed_timeout_s=120.0)
    assert config.timeout_s == 30.0

    # Invalid cases
    with pytest.raises(ValueError, match="must be positive"):
        OllamaConfig(timeout_s=0)

    with pytest.raises(ValueError, match="must be positive"):
        OllamaConfig(embed_timeout_s=-1.0)


def test_path_resolution():
    """Test that relative paths are resolved to absolute paths."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {
                "rag": {"index_path": "./code_index"},
                "tagging": {"cache_path": ".cache.db"},
            },
            f,
        )
        f.flush()

        try:
            app_config = load_config(f.name)
            # Paths should be resolved to absolute
            assert str(Path(app_config.rag.index_path).resolve()) == app_config.rag.index_path
            assert str(Path(app_config.tagging.cache_path).resolve()) == app_config.tagging.cache_path
        finally:
            Path(f.name).unlink()
