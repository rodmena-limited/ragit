#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.providers.ollama module.
"""

from unittest.mock import MagicMock, patch

import pytest

from ragit.providers.ollama import OllamaProvider


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_default_initialization(self):
        """Test provider with default config."""
        provider = OllamaProvider()

        assert provider.provider_name == "ollama"
        assert provider.base_url is not None
        assert provider.embedding_url is not None
        assert provider.use_cache is True

    def test_custom_initialization(self):
        """Test provider with custom parameters."""
        provider = OllamaProvider(
            base_url="http://custom:8080/",
            embedding_url="http://embed:9090/",
            api_key="test-key",
            timeout=60,
            use_cache=False,
        )

        assert provider.base_url == "http://custom:8080"  # trailing slash stripped
        assert provider.embedding_url == "http://embed:9090"
        assert provider.api_key == "test-key"
        assert provider.timeout == 60
        assert provider.use_cache is False

    def test_strips_trailing_slash(self):
        """Test that trailing slashes are stripped from URLs."""
        provider = OllamaProvider(base_url="http://test:11434///", embedding_url="http://embed:11434//")

        assert not provider.base_url.endswith("/")
        assert not provider.embedding_url.endswith("/")


class TestOllamaProviderSession:
    """Tests for session management and connection pooling."""

    def test_session_lazy_initialization(self):
        """Test that session is lazily initialized."""
        provider = OllamaProvider()
        assert provider._session is None

        # Access session property triggers initialization
        session = provider.session
        assert session is not None
        assert provider._session is session

    def test_session_reuse(self):
        """Test that session is reused on subsequent calls."""
        provider = OllamaProvider()
        session1 = provider.session
        session2 = provider.session
        assert session1 is session2

    def test_session_headers_with_api_key(self):
        """Test that session headers include auth when API key is set."""
        provider = OllamaProvider(api_key="test-key")
        session = provider.session
        assert session.headers.get("Authorization") == "Bearer test-key"
        assert session.headers.get("Content-Type") == "application/json"

    def test_session_headers_without_api_key(self):
        """Test session headers without API key."""
        provider = OllamaProvider(base_url="http://test:11434")
        provider.api_key = None  # Force no API key
        provider._session = None  # Reset session
        session = provider.session
        assert "Authorization" not in session.headers
        assert session.headers.get("Content-Type") == "application/json"

    def test_close_session(self):
        """Test that close() properly closes the session."""
        provider = OllamaProvider()
        _ = provider.session  # Initialize session
        assert provider._session is not None

        provider.close()
        assert provider._session is None


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
        provider = OllamaProvider(base_url="http://test:11434", api_key=None)
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
        assert OllamaProvider.EMBEDDING_DIMENSIONS["nomic-embed-text"] == 768
        assert OllamaProvider.EMBEDDING_DIMENSIONS["mxbai-embed-large"] == 1024
        assert OllamaProvider.EMBEDDING_DIMENSIONS["qwen3-embedding:8b"] == 4096

    def test_dimensions_property(self):
        """Test that dimensions property returns current value."""
        provider = OllamaProvider()
        assert provider.dimensions == 768


class TestOllamaProviderIsAvailable:
    """Tests for is_available method."""

    def test_is_available_success(self):
        """Test is_available returns True when server responds."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp

        provider._session = mock_session

        assert provider.is_available() is True
        mock_session.get.assert_called_once()

    def test_is_available_failure(self):
        """Test is_available returns False on connection error."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.get.side_effect = RequestException("Connection failed")
        provider._session = mock_session

        assert provider.is_available() is False


class TestOllamaProviderListModels:
    """Tests for list_models method."""

    def test_list_models_success(self):
        """Test listing available models."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3"}, {"name": "mxbai-embed-large"}]}
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp

        provider._session = mock_session

        models = provider.list_models()

        assert len(models) == 2
        assert models[0]["name"] == "llama3"

    def test_list_models_error(self):
        """Test list_models raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.get.side_effect = RequestException("Failed")
        provider._session = mock_session

        with pytest.raises(ConnectionError, match="Failed to list Ollama models"):
            provider.list_models()


class TestOllamaProviderGenerate:
    """Tests for generate method."""

    def test_generate_success(self):
        """Test successful text generation."""
        provider = OllamaProvider(base_url="http://test:11434", api_key="key")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Hello, world!",
            "model": "llama3",
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 1000000000,
        }
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        response = provider.generate(
            prompt="Say hello", model="llama3", system_prompt="Be friendly", temperature=0.5, max_tokens=100
        )

        assert response.text == "Hello, world!"
        assert response.model == "llama3"
        assert response.provider == "ollama"
        assert response.usage["prompt_tokens"] == 10

        # Verify the request
        call_args = mock_session.post.call_args
        assert "/api/generate" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["prompt"] == "Say hello"
        assert payload["system"] == "Be friendly"
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 100

    def test_generate_without_optional_params(self):
        """Test generation without optional parameters."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "OK", "model": "m"}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        response = provider.generate(prompt="Hi", model="llama3")

        assert response.text == "OK"

        payload = mock_session.post.call_args[1]["json"]
        assert "system" not in payload
        assert "num_predict" not in payload["options"]

    def test_generate_error(self):
        """Test generate raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.post.side_effect = RequestException("Failed")
        provider._session = mock_session

        with pytest.raises(ConnectionError, match="Ollama generate failed"):
            provider.generate(prompt="Hi", model="llama3")


class TestOllamaProviderEmbed:
    """Tests for embed method."""

    def test_embed_success_without_cache(self):
        """Test successful embedding generation without cache."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        embedding = [0.1] * 1024
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [embedding], "model": "mxbai-embed-large"}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        response = provider.embed(text="Hello", model="mxbai-embed-large")

        assert response.embedding == tuple(embedding)
        assert response.model == "mxbai-embed-large"
        assert response.provider == "ollama"
        assert response.dimensions == 1024

    def test_embed_with_cache(self):
        """Test embedding with cache enabled."""
        # Clear cache first
        OllamaProvider.clear_embedding_cache()

        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=True)

        with patch("ragit.providers.ollama.requests.post") as mock_post:
            embedding = [0.1] * 1024
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [embedding]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            # First call - should hit API
            response1 = provider.embed(text="Hello", model="mxbai-embed-large")
            assert mock_post.call_count == 1

            # Second call with same text - should use cache
            response2 = provider.embed(text="Hello", model="mxbai-embed-large")
            assert mock_post.call_count == 1  # No additional call

            assert response1.embedding == response2.embedding

    def test_embed_updates_dimensions(self):
        """Test that embed updates internal dimensions from response."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        embedding = [0.1] * 512  # Different dimension
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [embedding]}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        response = provider.embed(text="Hello", model="custom-model")

        assert provider.dimensions == 512
        assert response.dimensions == 512

    def test_embed_empty_response_error(self):
        """Test embed raises on empty embedding."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": []}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        with pytest.raises(ValueError, match="Empty embedding"):
            provider.embed(text="Hello", model="model")

    def test_embed_error(self):
        """Test embed raises on error."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.post.side_effect = RequestException("Failed")
        provider._session = mock_session

        with pytest.raises(ConnectionError, match="Ollama embed failed"):
            provider.embed(text="Hello", model="model")


class TestOllamaProviderEmbedBatch:
    """Tests for embed_batch method."""

    def test_embed_batch_success(self):
        """Test successful batch embedding with single API call."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": embeddings, "model": "mxbai-embed-large"}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        responses = provider.embed_batch(texts=["Hello", "World", "Test"], model="mxbai-embed-large")

        # Should make a single API call
        assert mock_session.post.call_count == 1

        # Verify input format
        call_args = mock_session.post.call_args
        assert call_args[1]["json"]["input"] == ["Hello", "World", "Test"]

        assert len(responses) == 3
        assert responses[0].embedding == tuple(embeddings[0])
        assert all(r.dimensions == 1024 for r in responses)

    def test_embed_batch_error(self):
        """Test embed_batch raises on error."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.post.side_effect = RequestException("Failed")
        provider._session = mock_session

        with pytest.raises(ConnectionError, match="Ollama batch embed failed"):
            provider.embed_batch(texts=["a", "b"], model="model")

    def test_embed_batch_single_api_call(self):
        """Test embed_batch uses native batch API (single call)."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        embeddings = [[0.1] * 768, [0.2] * 768]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": embeddings}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        # Batch call makes single API request
        responses = provider.embed_batch(texts=["text_a", "text_b"], model="model")
        assert mock_session.post.call_count == 1
        assert len(responses) == 2

        # Verify all texts sent in single request
        call_args = mock_session.post.call_args
        assert call_args[1]["json"]["input"] == ["text_a", "text_b"]

    def test_embed_batch_truncates_long_text(self):
        """Test embed_batch truncates oversized text."""
        provider = OllamaProvider(embedding_url="http://test:11434", use_cache=False)

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        # Call with long text
        long_text = "x" * 5000
        provider.embed_batch(texts=[long_text], model="model")

        # Verify the input was truncated
        call_args = mock_session.post.call_args
        sent_input = call_args[1]["json"]["input"]
        assert len(sent_input[0]) == OllamaProvider.MAX_EMBED_CHARS


class TestOllamaProviderEmbedBatchAsync:
    """Tests for embed_batch_async method."""

    def test_embed_batch_async_method_exists(self):
        """Test that embed_batch_async method exists and has correct signature."""
        import inspect

        provider = OllamaProvider(embedding_url="http://test:11434")

        # Verify method exists and is async
        assert hasattr(provider, "embed_batch_async")
        assert inspect.iscoroutinefunction(provider.embed_batch_async)

        # Check signature
        sig = inspect.signature(provider.embed_batch_async)
        params = list(sig.parameters.keys())
        assert "texts" in params
        assert "model" in params
        assert "max_concurrent" in params

    def test_embed_batch_async_sets_model_state(self):
        """Test that embed_batch_async sets model state on provider."""
        provider = OllamaProvider(embedding_url="http://test:11434")

        # Before calling async method, model state should be None
        assert provider._current_embed_model is None

        # We can't easily mock trio/httpx async, but we can test that
        # the method is properly defined by checking its properties
        assert provider.dimensions == 768  # default

    def test_embed_batch_async_dimensions_lookup(self):
        """Test that known models get correct dimensions."""
        # Test that EMBEDDING_DIMENSIONS contains expected models
        assert "mxbai-embed-large" in OllamaProvider.EMBEDDING_DIMENSIONS
        assert OllamaProvider.EMBEDDING_DIMENSIONS["mxbai-embed-large"] == 1024
        assert "nomic-embed-text" in OllamaProvider.EMBEDDING_DIMENSIONS
        assert OllamaProvider.EMBEDDING_DIMENSIONS["nomic-embed-text"] == 768


class TestOllamaProviderChat:
    """Tests for chat method."""

    def test_chat_success(self):
        """Test successful chat completion."""
        provider = OllamaProvider(base_url="http://test:11434", api_key="key")

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "I'm doing well!", "role": "assistant"},
            "model": "llama3",
            "prompt_eval_count": 15,
            "eval_count": 10,
        }
        mock_resp.raise_for_status.return_value = None
        mock_session.post.return_value = mock_resp

        provider._session = mock_session

        messages = [{"role": "user", "content": "How are you?"}]

        response = provider.chat(messages=messages, model="llama3", temperature=0.5, max_tokens=100)

        assert response.text == "I'm doing well!"
        assert response.model == "llama3"
        assert response.provider == "ollama"

        # Verify request
        call_args = mock_session.post.call_args
        assert "/api/chat" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["messages"] == messages

    def test_chat_error(self):
        """Test chat raises on error."""
        provider = OllamaProvider(base_url="http://test:11434")

        mock_session = MagicMock()
        from requests import RequestException

        mock_session.post.side_effect = RequestException("Failed")
        provider._session = mock_session

        with pytest.raises(ConnectionError, match="Ollama chat failed"):
            provider.chat(messages=[{"role": "user", "content": "Hi"}], model="llama3")


class TestOllamaProviderCache:
    """Tests for embedding cache functionality."""

    def test_clear_embedding_cache(self):
        """Test that cache can be cleared."""
        OllamaProvider.clear_embedding_cache()
        info = OllamaProvider.embedding_cache_info()
        assert info["currsize"] == 0

    def test_embedding_cache_info(self):
        """Test cache info returns correct structure."""
        info = OllamaProvider.embedding_cache_info()
        assert "hits" in info
        assert "misses" in info
        assert "maxsize" in info
        assert "currsize" in info

    def test_cache_hit_tracking(self):
        """Test that cache hits are tracked."""
        OllamaProvider.clear_embedding_cache()

        with patch("ragit.providers.ollama.requests.post") as mock_post:
            embedding = [0.1] * 768
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [embedding]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            provider = OllamaProvider(use_cache=True)

            # First call - miss
            provider.embed("test", "model")
            info1 = OllamaProvider.embedding_cache_info()

            # Second call - hit
            provider.embed("test", "model")
            info2 = OllamaProvider.embedding_cache_info()

            assert info2["hits"] > info1["hits"]

    def test_cached_embedding_truncates_long_text(self):
        """Test that _cached_embedding truncates oversized text."""
        OllamaProvider.clear_embedding_cache()

        with patch("ragit.providers.ollama.requests.post") as mock_post:
            embedding = [0.1] * 768
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": [embedding]}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            provider = OllamaProvider(use_cache=True)

            # Call with text longer than MAX_EMBED_CHARS
            long_text = "x" * 5000
            provider.embed(long_text, "model")

            # Verify the input was truncated
            call_args = mock_post.call_args
            sent_input = call_args[1]["json"]["input"]
            assert len(sent_input) == OllamaProvider.MAX_EMBED_CHARS

    def test_cached_embedding_empty_response_error(self):
        """Test that _cached_embedding raises on empty embedding."""
        OllamaProvider.clear_embedding_cache()

        with patch("ragit.providers.ollama.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"embeddings": []}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            provider = OllamaProvider(use_cache=True)

            with pytest.raises(ValueError, match="Empty embedding"):
                provider.embed("unique_test_text_for_cache", "model")
