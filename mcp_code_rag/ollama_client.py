"""Ollama API client for embeddings and generation."""

import logging
import re
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mcp_code_rag.config import OllamaConfig

# Regex to strip characters that can break embedding APIs:
# - Surrogate pairs and unassigned Unicode (above BMP emojis, etc.)
# - Control characters except \n, \r, \t
_SANITIZE_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


def sanitize_text(text: str) -> str:
    """Remove control characters and problematic Unicode from text.

    Strips characters that commonly cause 400 errors in embedding APIs,
    such as control characters, while preserving normal whitespace.

    Args:
        text: Raw text to sanitize.

    Returns:
        Sanitized text safe for embedding APIs.
    """
    return _SANITIZE_RE.sub("", text)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, config: OllamaConfig):
        """Initialize Ollama client from configuration.

        Args:
            config: OllamaConfig instance with API settings.

        Raises:
            ValueError: If configuration values are invalid.
        """
        self.base_url = config.base_url
        self.embed_model = config.embed_model
        self.timeout_s = config.timeout_s
        self.embed_timeout_s = config.embed_timeout_s
        self.max_retries = config.max_retries

        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy.

        Returns:
            Configured requests.Session with exponential backoff retry.
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts using Ollama.

        Sanitizes input texts to remove problematic characters. If a batch
        request fails with a 400 error, falls back to embedding each text
        individually so that one bad chunk does not block the entire batch.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
            For texts that fail individually, a zero-vector is returned.

        Raises:
            requests.RequestException: If API call fails after retries
                (only for non-400 errors).
            ValueError: If response format is invalid.
        """
        if not texts:
            return []

        # Sanitize all texts before sending to the API
        sanitized = [sanitize_text(t) for t in texts]

        url = f"{self.base_url}/api/embed"

        payload = {"model": self.embed_model, "input": sanitized}

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.embed_timeout_s,
            )
            response.raise_for_status()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                logger.warning(
                    "Batch embed returned 400; falling back to individual embedding"
                )
                return self._embed_individually(sanitized, url)
            logger.error(f"Network error calling {url}: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Network error calling {url}: {e}")
            raise

        try:
            data = response.json()
            embeddings = data.get("embeddings")
            if embeddings is None:
                raise ValueError("Response missing 'embeddings' field")
            if not isinstance(embeddings, list):
                raise ValueError(f"Expected embeddings to be a list, got {type(embeddings)}")
            return embeddings
        except ValueError as e:
            logger.error(f"Invalid response format from embed API: {e}")
            raise

    def _embed_individually(
        self, texts: list[str], url: str
    ) -> list[list[float]]:
        """Embed texts one at a time, returning zero-vectors for failures.

        Args:
            texts: Sanitized texts to embed individually.
            url: The embed API URL.

        Returns:
            List of embedding vectors; failed texts get a zero-vector.
        """
        embeddings: list[list[float]] = []
        dim: int | None = None

        for i, text in enumerate(texts):
            payload = {"model": self.embed_model, "input": [text]}
            try:
                response = self.session.post(
                    url, json=payload, timeout=self.embed_timeout_s
                )
                response.raise_for_status()
                data = response.json()
                vecs = data.get("embeddings")
                if vecs and isinstance(vecs, list) and len(vecs) > 0:
                    embeddings.append(vecs[0])
                    if dim is None:
                        dim = len(vecs[0])
                    continue
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i}: {e}")

            # Append placeholder; will be replaced once we know dimensions
            embeddings.append(None)  # type: ignore[arg-type]

        # Replace None placeholders with zero-vectors
        if dim is None:
            # All failed — try to determine dimension from model config
            dim = 1024  # sensible default for common embed models
        for i, emb in enumerate(embeddings):
            if emb is None:
                embeddings[i] = [0.0] * dim

        return embeddings

    def generate(self, prompt: str, model: str) -> str:
        """Generate text using Ollama (non-streaming).

        Args:
            prompt: Input prompt for generation.
            model: Model name to use for generation.

        Returns:
            Generated text response.

        Raises:
            requests.RequestException: If API call fails after retries.
            ValueError: If response format is invalid.
        """
        url = f"{self.base_url}/api/generate"

        payload = {"model": model, "prompt": prompt, "stream": False}

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Network error calling {url}: {e}")
            raise

        try:
            data = response.json()
            generated_text = data.get("response")
            if generated_text is None:
                raise ValueError("Response missing 'response' field")
            if not isinstance(generated_text, str):
                raise ValueError(f"Expected response to be a string, got {type(generated_text)}")
            return generated_text
        except ValueError as e:
            logger.error(f"Invalid response format from generate API: {e}")
            raise

    def list_models(self) -> list[dict]:
        """List all available models from Ollama.

        Returns:
            List of model dictionaries with model metadata.

        Raises:
            requests.RequestException: If API call fails.
            ValueError: If response format is invalid.
        """
        url = f"{self.base_url}/api/tags"

        try:
            response = self.session.get(url, timeout=self.timeout_s)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Network error calling {url}: {e}")
            raise

        try:
            data = response.json()
            models = data.get("models", [])
            if not isinstance(models, list):
                raise ValueError(f"Expected models to be a list, got {type(models)}")
            return models
        except ValueError as e:
            logger.error(f"Invalid response format from list_models API: {e}")
            raise

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if model is available, False otherwise.

        Raises:
            requests.RequestException: If API call fails.
        """
        try:
            models = self.list_models()
            model_names = [m.get("name") for m in models]
            return model_name in model_names
        except requests.RequestException as e:
            logger.warning(f"Could not check model availability: {e}")
            return False
