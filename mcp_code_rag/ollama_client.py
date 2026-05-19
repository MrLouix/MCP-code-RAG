"""Ollama API client for embeddings and generation."""

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from mcp_code_rag.config import OllamaConfig

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

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            requests.RequestException: If API call fails after retries.
            ValueError: If response format is invalid.
        """
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"

        payload = {"model": self.embed_model, "input": texts}

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.embed_timeout_s,
            )
            response.raise_for_status()
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
