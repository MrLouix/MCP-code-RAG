"""LLM-based tagging (H2) for source code files."""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from mcp_code_rag.config import OllamaConfig, TaggingConfig
from mcp_code_rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class LLMTagger:
    """LLM-based tagger for enhanced code classification."""

    def __init__(
        self,
        config: TaggingConfig,
        ollama_client: OllamaClient,
        ollama_config: Optional[OllamaConfig] = None,
    ):
        """Initialize LLMTagger with configuration and Ollama client.

        Args:
            config: TaggingConfig instance.
            ollama_client: OllamaClient for generation.
            ollama_config: OllamaConfig instance. If None, creates default.
        """
        self.config = config
        self.ollama_client = ollama_client
        self.ollama_config = ollama_config or OllamaConfig()
        self.enabled = config.h2_enabled
        self.cache_path = Path(config.cache_path)

        # Initialize cache database if enabled
        if config.use_cache:
            self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initialize SQLite cache database."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tag_cache (
                        file_hash TEXT PRIMARY KEY,
                        tags TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Failed to initialize cache database: {e}")

    def _compute_file_hash(self, file_path: str, source_preview: str) -> str:
        """Compute SHA256 hash of file path and content preview.

        Args:
            file_path: Path to the file.
            source_preview: Preview of file content.

        Returns:
            Hex-encoded SHA256 hash.
        """
        content = f"{file_path}:{source_preview}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_cached_tags(self, file_hash: str) -> Optional[list[str]]:
        """Retrieve cached tags for a file hash.

        Args:
            file_hash: SHA256 hash of file.

        Returns:
            List of tags if cached, None otherwise.
        """
        if not self.config.use_cache:
            return None

        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    "SELECT tags FROM tag_cache WHERE file_hash = ?",
                    (file_hash,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except sqlite3.Error as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def cache_tags(self, file_hash: str, tags: list[str]) -> None:
        """Cache tags for a file hash.

        Args:
            file_hash: SHA256 hash of file.
            tags: List of tags to cache.
        """
        if not self.config.use_cache:
            return

        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO tag_cache (file_hash, tags) VALUES (?, ?)",
                    (file_hash, json.dumps(tags)),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.warning(f"Cache write error: {e}")

    def _validate_tags(self, tags: list[str]) -> list[str]:
        """Validate tags against taxonomy.

        Args:
            tags: List of tags to validate.

        Returns:
            Filtered list of valid tags.
        """
        valid_tags = set()
        taxonomy = self.config.taxonomy

        for tag in tags:
            if ":" not in tag:
                continue

            category, value = tag.split(":", 1)
            is_valid = False

            if category == "framework" and value in taxonomy.framework:
                is_valid = True
            elif category == "design_pattern" and value in taxonomy.design_pattern:
                is_valid = True
            elif category == "purpose" and value in ["api", "util", "test", "doc"]:
                # purpose values are open-ended but with common patterns
                is_valid = True
            elif category in ["lang", "type", "layer"]:
                # These are typically validated by heuristics, accept them
                is_valid = True

            if is_valid:
                valid_tags.add(tag)
            else:
                logger.debug(f"Rejected invalid tag: {tag}")

        return sorted(list(valid_tags))

    def _build_prompt(
        self,
        file_path: str,
        source_preview: str,
        imports: list[str],
        symbols: list[str],
    ) -> str:
        """Build prompt for LLM tagging.

        Args:
            file_path: Path to the file.
            source_preview: First 50 lines or so of source.
            imports: List of imported modules/libraries.
            symbols: List of defined symbols (functions, classes).

        Returns:
            Formatted prompt string.
        """
        prompt = f"""Analyze this code file and provide tags in JSON format.

FILE: {file_path}

IMPORTS:
{chr(10).join(f"- {imp}" for imp in imports) if imports else "None"}

SYMBOLS (functions/classes):
{chr(10).join(f"- {sym}" for sym in symbols) if symbols else "None"}

SOURCE (first 50 lines):
{source_preview}

Respond with a JSON object with a "tags" key containing an array of tags.
Tags should be in format: "category:value"
Categories: framework, design_pattern, purpose
Example: {{"tags": ["framework:fastapi", "design_pattern:repository", "purpose:api"]}}

IMPORTANT: Only use tags from these taxonomies:
Frameworks: {", ".join(self.config.taxonomy.framework)}
Design Patterns: {", ".join(self.config.taxonomy.design_pattern)}
Purposes: api, util, test, doc

Respond with ONLY the JSON object, no other text."""
        return prompt

    def tag_h2(
        self,
        file_path: str,
        source_preview: str,
        imports: list[str],
        symbols: list[str],
    ) -> list[str]:
        """Apply LLM-based tagging (H2) to a file.

        Args:
            file_path: Path to the file.
            source_preview: Preview of source code (first ~50 lines).
            imports: List of imported modules.
            symbols: List of defined symbols.

        Returns:
            List of LLM-generated tags (validated against taxonomy).
            Empty list if h2_enabled is False.

        Raises:
            Exception: If LLM generation fails (only if h2_enabled is True).
        """
        if not self.enabled:
            return []

        file_hash = self._compute_file_hash(file_path, source_preview)

        # Check cache
        cached_tags = self.get_cached_tags(file_hash)
        if cached_tags is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached_tags

        # Build and execute prompt
        prompt = self._build_prompt(file_path, source_preview, imports, symbols)

        try:
            response = self.ollama_client.generate(
                prompt=prompt,
                model=self.ollama_config.tag_model,
            )
            tags = self._parse_response(response)
            validated_tags = self._validate_tags(tags)
            self.cache_tags(file_hash, validated_tags)
            return validated_tags
        except Exception as e:
            logger.error(f"LLM tagging failed for {file_path}: {e}")
            raise

    def _parse_response(self, response: str) -> list[str]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response from LLM.

        Returns:
            List of tags extracted from JSON.

        Raises:
            ValueError: If JSON is invalid or missing tags key.
        """
        try:
            # Try to extract JSON from response
            response = response.strip()
            data = json.loads(response)

            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")

            tags = data.get("tags", [])
            if not isinstance(tags, list):
                raise ValueError("tags field is not a list")

            return [str(tag).strip() for tag in tags]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
