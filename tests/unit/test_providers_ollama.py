#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.providers.ollama module.
"""

import pytest
from unittest.mock import patch, MagicMock
from requests import HTTPError

from ragit.providers.ollama import OllamaProvider


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_default_initialization(self):
        """Test provider with default config."""
        provider = OllamaProvider()

        assert provider.provider_name == "ollama"
        assert provider.base_url is not None
        assert provider.embedding_url is not None

    def test_custom_initialization(self):
        """Test provider with custom parameters."""
        provider = OllamaProvider(
            base_url="http://custom:8080/",
            embedding_url="http://embed:9090/",
            api_key="test-key",
            timeout=60
        )

        assert provider.base_url == "http://custom:8080"  # trailing slash stripped
        assert provider.embedding_url == "http://embed:9090"
        assert provider.api_key == "test-key"
        assert provider.timeout == 60

    def test_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from URLs."""
        provider = OllamaProvider(
            base_url="http://test:11434///",
            embedding_url="http://embed:11434//"
        )

        assert not provider.base_url.endswith("/")
        assert not provider.embedding_url.endswith("/")


class TestOllamaProviderHeaders:
    """Tests for header generation."""

    def test_headers_with_api_key(self):
        """Test that auth header is included when API key is set."""
        provider = OllamaProvider(api_key="secret-key")
        headers = provider._get_headers(include_auth=True)

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer secret-key"

    def test_headers_without_api_key(self):
        """Test headers when no API key is set."""
        # Explicitly set api_key to None to override any env var
        provider = OllamaProvider(
            base_url="http://test:11434",
            api_key=None
        )
        # Force api_key to None (in case config loaded one from env)
        provider.api_key = None
        headers = provider._get_headers(include_auth=True)

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_auth_disabled(self):
        """Test that auth can be disabled even with API key."""
        provider = OllamaProvider(api_key="secret-key")
        headers = provider._get_headers(include_auth=False)

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers


class TestOllamaProviderDimensions:
    """Tests for embedding dimensions."""

    def test_known_model_dimensions(self):
        """Test dimensions for known embedding models."""
        provider = OllamaProvider()

        # Access the class-level constant
        assert OllamaProvider.EMBEDDING_DIMENSIONS["nomic-embed-text"] == 768
        assert OllamaProvider.EMBEDDING_DIMENSIONS["mxbai-embed-large"] == 1024
        assert OllamaProvider.EMBEDDING_DIMENSIONS["qwen3-embedding:8b"] == 4096

    def test_dimensions_property(self):
        """Test that dimensions property returns current value."""
        provider = OllamaProvider()

        # Default dimension
        assert provider.dimensions == 768


class TestOllamaProviderIsAvailable:
    """Tests for is_available method."""

    def test_is_available_success(self):
        """Test is_available returns True when server responds."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            assert provider.is_available() is True
            mock_get.assert_called_once()

    def test_is_available_failure(self):
        """Test is_available returns False on connection error."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.get") as mock_get:
            from requests import RequestException
            mock_get.side_effect = RequestException("Connection failed")

            assert provider.is_available() is False


class TestOllamaProviderListModels:
    """Tests for list_models method."""

    def test_list_models_success(self):
        """Test listing available models."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "models": [
                    {"name": "llama3"},
                    {"name": "mxbai-embed-large"}
                ]
            }
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            models = provider.list_models()

            assert len(models) == 2
            assert models[0]["name"] == "llama3"

    def test_list_models_error(self):
        """Test list_models raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.get") as mock_get:
            from requests import RequestException
            mock_get.side_effect = RequestException("Failed")

            with pytest.raises(ConnectionError, match="Failed to list Ollama models"):
                provider.list_models()


class TestOllamaProviderGenerate:
    """Tests for generate method."""

    def test_generate_success(self):
        """Test successful text generation."""
        provider = OllamaProvider(base_url="http://test:11434", api_key="key")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "response": "Hello, world!",
                "model": "llama3",
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 1000000000
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            response = provider.generate(
                prompt="Say hello",
                model="llama3",
                system_prompt="Be friendly",
                temperature=0.5,
                max_tokens=100
            )

            assert response.text == "Hello, world!"
            assert response.model == "llama3"
            assert response.provider == "ollama"
            assert response.usage["prompt_tokens"] == 10

            # Verify the request
            call_args = mock_post.call_args
            assert "/api/generate" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["prompt"] == "Say hello"
            assert payload["system"] == "Be friendly"
            assert payload["options"]["temperature"] == 0.5
            assert payload["options"]["num_predict"] == 100

    def test_generate_without_optional_params(self):
        """Test generation without optional parameters."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "OK", "model": "m"}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            response = provider.generate(prompt="Hi", model="llama3")

            assert response.text == "OK"

            payload = mock_post.call_args[1]["json"]
            assert "system" not in payload
            assert "num_predict" not in payload["options"]

    def test_generate_error(self):
        """Test generate raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.post") as mock_post:
            from requests import RequestException
            mock_post.side_effect = RequestException("Failed")

            with pytest.raises(ConnectionError, match="Ollama generate failed"):
                provider.generate(prompt="Hi", model="llama3")


class TestOllamaProviderEmbed:
    """Tests for embed method."""

    def test_embed_success(self):
        """Test successful embedding generation."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            embedding = [0.1] * 1024
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": [embedding],
                "model": "mxbai-embed-large"
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            response = provider.embed(text="Hello", model="mxbai-embed-large")

            assert response.embedding == embedding
            assert response.model == "mxbai-embed-large"
            assert response.provider == "ollama"
            assert response.dimensions == 1024

            # Verify embedding URL used and no auth
            call_args = mock_post.call_args
            assert "/api/embed" in call_args[0][0]
            assert "Authorization" not in call_args[1]["headers"]

    def test_embed_updates_dimensions(self):
        """Test that embed updates internal dimensions from response."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            embedding = [0.1] * 512  # Different dimension
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [embedding]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            response = provider.embed(text="Hello", model="custom-model")

            assert provider.dimensions == 512
            assert response.dimensions == 512

    def test_embed_empty_response_error(self):
        """Test embed raises on empty embedding."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [[]]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            with pytest.raises(ValueError, match="Empty embedding"):
                provider.embed(text="Hello", model="model")

    def test_embed_error(self):
        """Test embed raises on error."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            from requests import RequestException
            mock_post.side_effect = RequestException("Failed")

            with pytest.raises(ConnectionError, match="Ollama embed failed"):
                provider.embed(text="Hello", model="model")


class TestOllamaProviderEmbedBatch:
    """Tests for embed_batch method."""

    def test_embed_batch_success(self):
        """Test successful batch embedding."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": embeddings,
                "model": "mxbai-embed-large"
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            responses = provider.embed_batch(
                texts=["Hello", "World", "Test"],
                model="mxbai-embed-large"
            )

            assert len(responses) == 3
            assert responses[0].embedding == embeddings[0]
            assert all(r.dimensions == 1024 for r in responses)

    def test_embed_batch_error(self):
        """Test embed_batch raises on error."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        with patch("requests.post") as mock_post:
            from requests import RequestException
            mock_post.side_effect = RequestException("Failed")

            with pytest.raises(ConnectionError, match="Ollama batch embed failed"):
                provider.embed_batch(texts=["a", "b"], model="model")


class TestOllamaProviderChat:
    """Tests for chat method."""

    def test_chat_success(self):
        """Test successful chat completion."""
        provider = OllamaProvider(base_url="http://test:11434", api_key="key")

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "message": {"content": "I'm doing well!", "role": "assistant"},
                "model": "llama3",
                "prompt_eval_count": 15,
                "eval_count": 10
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            messages = [
                {"role": "user", "content": "How are you?"}
            ]

            response = provider.chat(
                messages=messages,
                model="llama3",
                temperature=0.5,
                max_tokens=100
            )

            assert response.text == "I'm doing well!"
            assert response.model == "llama3"
            assert response.provider == "ollama"

            # Verify request
            call_args = mock_post.call_args
            assert "/api/chat" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["messages"] == messages

    def test_chat_error(self):
        """Test chat raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        with patch("requests.post") as mock_post:
            from requests import RequestException
            mock_post.side_effect = RequestException("Failed")

            with pytest.raises(ConnectionError, match="Ollama chat failed"):
                provider.chat(messages=[{"role": "user", "content": "Hi"}], model="llama3")
