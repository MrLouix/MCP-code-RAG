"""Configuration management for MCP Code RAG using Pydantic models."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    """Ollama API configuration."""

    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    tag_model: str = "qwen3.5"
    timeout_s: float = 30.0
    embed_timeout_s: float = 120.0
    max_retries: int = 3
    auto_pull: bool = False

    @field_validator("timeout_s", "embed_timeout_s")
    @classmethod
    def positive_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @field_validator("max_retries")
    @classmethod
    def non_negative_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v


class RagConfig(BaseModel):
    """RAG (Retrieval-Augmented Generation) configuration."""

    index_path: str = "./code_rag_index"
    chunk_strategy: str = "ast"
    max_embed_text_length: int = 8000
    max_chunks_per_doc: int = 500
    supported_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".mjs",
            ".ts",
            ".tsx",
            ".go",
            ".cs",
            ".rs",
            ".java",
            ".rb",
            ".php",
            ".c",
            ".h",
            ".cpp",
            ".swift",
            ".kt",
            ".md",
            ".markdown",
            ".txt",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".cfg",
            ".ini",
            ".env",
            ".xml",
        ]
    )
    default_exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "*.min.js",
            "*.min.css",
            "*.lock",
            ".DS_Store",
            "*.egg-info",
            ".vscode-server",
            ".claude",
            ".mypy_cache",
            ".ruff_cache",
        ]
    )

    @field_validator("chunk_strategy")
    @classmethod
    def valid_chunk_strategy(cls, v: str) -> str:
        valid_strategies = ["ast", "regex", "text"]
        if v not in valid_strategies:
            raise ValueError(f"chunk_strategy must be one of {valid_strategies}")
        return v

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve relative paths to absolute paths."""
        if not Path(self.index_path).is_absolute():
            self.index_path = str(base_dir / self.index_path)


class HybridSearchConfig(BaseModel):
    """Hybrid search (semantic + keyword) configuration."""

    enabled: bool = True
    alpha: float = 0.6
    beta: float = 0.4
    fts5_table: str = "code_fts"

    @field_validator("alpha", "beta")
    @classmethod
    def valid_weights(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("alpha and beta must be between 0 and 1")
        return v

    @field_validator("beta")
    @classmethod
    def weights_sum_to_one(cls, v: float, info) -> float:
        if "alpha" in info.data:
            alpha = info.data["alpha"]
            if not abs(alpha + v - 1.0) < 1e-6:
                raise ValueError(f"alpha + beta must equal 1.0, got {alpha} + {v}")
        return v


class TaxonomyConfig(BaseModel):
    """Taxonomy configuration for tagging."""

    lang: list[str] = Field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "go",
            "csharp",
            "rust",
            "java",
            "ruby",
            "php",
        ]
    )
    framework: list[str] = Field(
        default_factory=lambda: [
            "fastapi",
            "django",
            "flask",
            "express",
            "nextjs",
            "react",
            "gin",
            "aspnet",
        ]
    )
    layer: list[str] = Field(
        default_factory=lambda: [
            "api",
            "model",
            "service",
            "utils",
            "config",
            "test",
            "middleware",
        ]
    )
    design_pattern: list[str] = Field(
        default_factory=lambda: [
            "repository",
            "factory",
            "singleton",
            "adapter",
            "observer",
            "middleware",
        ]
    )


class TaggingConfig(BaseModel):
    """Tagging configuration."""

    auto_tag_enabled: bool = True
    h2_enabled: bool = False
    timeout_ms: int = 30000
    use_cache: bool = True
    cache_path: str = ".code_tag_cache.db"
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)

    @field_validator("timeout_ms")
    @classmethod
    def positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout_ms must be positive")
        return v

    def resolve_paths(self, base_dir: Path) -> None:
        """Resolve relative paths to absolute paths."""
        if not Path(self.cache_path).is_absolute():
            self.cache_path = str(base_dir / self.cache_path)


class WatcherConfig(BaseModel):
    """File watcher configuration."""

    enabled: bool = False
    debounce_ms: int = 3000
    sync_deletions: bool = True
    max_workers: int = 4
    default_watch_paths: list[str] = Field(default_factory=list)
    default_recursive: bool = True
    exclude_patterns: list[str] = Field(default_factory=list)

    @field_validator("debounce_ms", "max_workers")
    @classmethod
    def positive_values(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("debounce_ms and max_workers must be positive")
        return v


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    collection_prefix: str = "coderag"
    auto_detect: bool = True
    cache_enabled: bool = True


class SecurityConfig(BaseModel):
    """Security configuration."""

    max_file_size_mb: float = 10.0
    regex_timeout_ms: int = 200
    yaml_safe_load_only: bool = True

    @field_validator("max_file_size_mb")
    @classmethod
    def positive_file_size(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_file_size_mb must be positive")
        return v

    @field_validator("regex_timeout_ms")
    @classmethod
    def positive_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("regex_timeout_ms must be positive")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "text"
    file: str = ""

    @field_validator("level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        valid_formats = ["text", "json"]
        if v not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        return v


class AppConfig(BaseModel):
    """Root application configuration aggregating all sections."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    tagging: TaggingConfig = Field(default_factory=TaggingConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def resolve_paths(self, base_dir: Optional[Path] = None) -> None:
        """Resolve all relative paths to absolute paths."""
        if base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(base_dir)

        self.rag.resolve_paths(base_dir)
        self.tagging.resolve_paths(base_dir)


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file with fallback to defaults.

    Args:
        path: Path to configuration file. If None, uses config.yaml in current directory.

    Returns:
        AppConfig instance with loaded or default configuration.

    Raises:
        FileNotFoundError: If specified path does not exist.
        yaml.YAMLError: If YAML is malformed.
        ValueError: If configuration validation fails.
    """
    config_path = Path(path) if path else Path("config.yaml")

    if not config_path.exists():
        if path:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        # Use defaults if config.yaml not found and no path specified
        config_data = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # Create AppConfig from loaded data (missing sections use defaults)
    app_config = AppConfig(**config_data)

    # Resolve relative paths to absolute
    app_config.resolve_paths(config_path.parent if config_path.exists() else None)

    return app_config
