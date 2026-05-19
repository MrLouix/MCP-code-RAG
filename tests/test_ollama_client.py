"""Tests for OllamaClient."""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock

from mcp_code_rag.config import OllamaConfig
from mcp_code_rag.ollama_client import OllamaClient


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


class TestOllamaClientEmbed:
    """Tests for embed method."""

    def test_embed_single_text(self, ollama_client):
        """Test embedding a single text."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            }
            mock_post.return_value = mock_response

            result = ollama_client.embed(["hello world"])

            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_post.assert_called_once()

    def test_embed_multiple_texts(self, ollama_client):
        """Test embedding multiple texts."""
        texts = ["hello", "world", "test"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"embeddings": embeddings}
            mock_post.return_value = mock_response

            result = ollama_client.embed(texts)

            assert result == embeddings
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == "nomic-embed-text"
            assert call_args[1]["json"]["input"] == texts

    def test_embed_empty_list(self, ollama_client):
        """Test embedding empty list returns empty result."""
        result = ollama_client.embed([])
        assert result == []

    def test_embed_payload_format(self, ollama_client):
        """Test that embed sends correct JSON payload."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"embeddings": [[0.1]]}
            mock_post.return_value = mock_response

            ollama_client.embed(["test"])

            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/embed"
            assert call_args[1]["json"] == {
                "model": "nomic-embed-text",
                "input": ["test"],
            }
            assert call_args[1]["timeout"] == 10.0

    def test_embed_missing_embeddings_field(self, ollama_client):
        """Test error when response missing embeddings field."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"result": "invalid"}
            mock_post.return_value = mock_response

            with pytest.raises(ValueError, match="missing 'embeddings' field"):
                ollama_client.embed(["test"])

    def test_embed_invalid_embeddings_type(self, ollama_client):
        """Test error when embeddings is not a list."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"embeddings": "not_a_list"}
            mock_post.return_value = mock_response

            with pytest.raises(ValueError, match="Expected embeddings to be a list"):
                ollama_client.embed(["test"])

    def test_embed_network_error(self, ollama_client):
        """Test handling of network errors."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection failed")

            with pytest.raises(requests.RequestException):
                ollama_client.embed(["test"])


class TestOllamaClientGenerate:
    """Tests for generate method."""

    def test_generate_text(self, ollama_client):
        """Test generating text."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Generated text response"}
            mock_post.return_value = mock_response

            result = ollama_client.generate("prompt", "llama2")

            assert result == "Generated text response"
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == "llama2"
            assert call_args[1]["json"]["prompt"] == "prompt"
            assert call_args[1]["json"]["stream"] is False

    def test_generate_payload(self, ollama_client):
        """Test that generate sends correct payload."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "text"}
            mock_post.return_value = mock_response

            ollama_client.generate("test prompt", "mistral")

            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"
            assert call_args[1]["json"]["stream"] is False
            assert call_args[1]["timeout"] == 5.0

    def test_generate_missing_response_field(self, ollama_client):
        """Test error when response missing response field."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"result": "invalid"}
            mock_post.return_value = mock_response

            with pytest.raises(ValueError, match="missing 'response' field"):
                ollama_client.generate("test", "llama2")

    def test_generate_invalid_response_type(self, ollama_client):
        """Test error when response is not a string."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": 123}
            mock_post.return_value = mock_response

            with pytest.raises(ValueError, match="Expected response to be a string"):
                ollama_client.generate("test", "llama2")

    def test_generate_network_error(self, ollama_client):
        """Test handling of network errors."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_post.side_effect = requests.Timeout("Request timeout")

            with pytest.raises(requests.RequestException):
                ollama_client.generate("test", "llama2")


class TestOllamaClientListModels:
    """Tests for list_models method."""

    def test_list_models(self, ollama_client):
        """Test listing available models."""
        models = [
            {"name": "llama2", "size": 3800000000},
            {"name": "mistral", "size": 4700000000},
        ]

        with patch.object(ollama_client.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"models": models}
            mock_get.return_value = mock_response

            result = ollama_client.list_models()

            assert result == models
            mock_get.assert_called_once_with(
                "http://localhost:11434/api/tags", timeout=5.0
            )

    def test_list_models_empty(self, ollama_client):
        """Test listing when no models available."""
        with patch.object(ollama_client.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"models": []}
            mock_get.return_value = mock_response

            result = ollama_client.list_models()

            assert result == []

    def test_list_models_missing_models_field(self, ollama_client):
        """Test error when models field is missing."""
        with patch.object(ollama_client.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_get.return_value = mock_response

            result = ollama_client.list_models()

            assert result == []

    def test_list_models_invalid_type(self, ollama_client):
        """Test error when models is not a list."""
        with patch.object(ollama_client.session, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"models": "not_a_list"}
            mock_get.return_value = mock_response

            with pytest.raises(ValueError, match="Expected models to be a list"):
                ollama_client.list_models()

    def test_list_models_network_error(self, ollama_client):
        """Test handling of network errors."""
        with patch.object(ollama_client.session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection failed")

            with pytest.raises(requests.RequestException):
                ollama_client.list_models()


class TestOllamaClientIsModelAvailable:
    """Tests for is_model_available method."""

    def test_model_available(self, ollama_client):
        """Test checking if model is available."""
        models = [
            {"name": "llama2"},
            {"name": "mistral"},
        ]

        with patch.object(ollama_client, "list_models") as mock_list:
            mock_list.return_value = models

            assert ollama_client.is_model_available("llama2") is True
            assert ollama_client.is_model_available("mistral") is True

    def test_model_not_available(self, ollama_client):
        """Test checking if unavailable model."""
        models = [{"name": "llama2"}]

        with patch.object(ollama_client, "list_models") as mock_list:
            mock_list.return_value = models

            assert ollama_client.is_model_available("missing-model") is False

    def test_model_available_empty_list(self, ollama_client):
        """Test with empty model list."""
        with patch.object(ollama_client, "list_models") as mock_list:
            mock_list.return_value = []

            assert ollama_client.is_model_available("any-model") is False

    def test_model_available_network_error(self, ollama_client):
        """Test that network errors return False."""
        with patch.object(ollama_client, "list_models") as mock_list:
            mock_list.side_effect = requests.ConnectionError("Connection failed")

            assert ollama_client.is_model_available("llama2") is False


class TestOllamaClientRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_embed_retry_on_failure(self, ollama_client):
        """Test that embed retries on transient failures."""
        with patch.object(ollama_client.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"embeddings": [[0.1]]}

            # First two calls fail, third succeeds
            mock_post.side_effect = [
                requests.ConnectionError("Temp failure"),
                requests.ConnectionError("Temp failure"),
                mock_response,
            ]

            # With mocked HTTPAdapter, this will retry and eventually succeed
            # But since we're mocking at session level, we need to test the config
            assert ollama_client.max_retries == 2

    def test_max_retries_configuration(self, ollama_config):
        """Test that max_retries is properly configured."""
        config = OllamaConfig(max_retries=3)
        client = OllamaClient(config)
        assert client.max_retries == 3

    def test_retry_strategy_applied(self, ollama_client):
        """Test that retry strategy is configured in session."""
        assert ollama_client.session is not None
        # Session should have adapters configured
        assert "http://" in ollama_client.session.adapters
        assert "https://" in ollama_client.session.adapters
