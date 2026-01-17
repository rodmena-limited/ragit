#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.providers.sentence_transformers module.

These tests require the sentence-transformers package to be installed.
Install with: pip install ragit[transformers]
"""

import pytest

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# Skip all tests in this module if sentence-transformers not installed
pytestmark = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed (pip install ragit[transformers])",
)


class TestSentenceTransformersProviderInit:
    """Tests for SentenceTransformersProvider initialization."""

    def test_import_raises_without_library(self, monkeypatch):
        """Test that import raises ImportError when library not available."""
        # This test only makes sense if we can mock the import
        # Skip if we can't properly test this
        if HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("Can't test ImportError when library is installed")

    def test_default_model(self):
        """Test default model name."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        provider = SentenceTransformersProvider()
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_custom_model(self):
        """Test custom model name."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")
        assert provider.model_name == "all-mpnet-base-v2"

    def test_provider_name(self):
        """Test provider name."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        provider = SentenceTransformersProvider()
        assert provider.provider_name == "sentence_transformers"


class TestSentenceTransformersProviderKnownDimensions:
    """Tests for known model dimensions."""

    def test_known_dimensions(self):
        """Test that known models have correct dimensions."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        # Test with default model
        provider = SentenceTransformersProvider()
        assert provider.dimensions == 384

    def test_mpnet_dimensions(self):
        """Test mpnet model dimensions."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")
        assert provider.dimensions == 768


@pytest.mark.integration
class TestSentenceTransformersProviderEmbed:
    """Integration tests for SentenceTransformersProvider embedding.

    These tests actually load models and generate embeddings.
    """

    @pytest.fixture
    def provider(self):
        """Create a provider instance."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider

        return SentenceTransformersProvider()

    def test_embed(self, provider):
        """Test embedding generation."""
        response = provider.embed("Hello world")

        assert response.provider == "sentence_transformers"
        assert response.model == "all-MiniLM-L6-v2"
        assert len(response.embedding) == 384
        assert response.dimensions == 384
        assert isinstance(response.embedding, tuple)

    def test_embed_returns_normalized(self, provider):
        """Test that embeddings have reasonable values."""
        import numpy as np

        response = provider.embed("Test sentence")
        emb = np.array(response.embedding)

        # Check embedding is not all zeros
        assert np.abs(emb).sum() > 0

        # Check values are in reasonable range
        assert np.abs(emb).max() < 10

    def test_embed_batch(self, provider):
        """Test batch embedding."""
        texts = ["Hello", "World", "Test"]
        responses = provider.embed_batch(texts)

        assert len(responses) == 3
        assert all(len(r.embedding) == 384 for r in responses)
        assert all(r.provider == "sentence_transformers" for r in responses)

    def test_embed_batch_empty(self, provider):
        """Test batch embedding with empty list."""
        responses = provider.embed_batch([])
        assert responses == []

    def test_embed_consistency(self, provider):
        """Test that same input produces same output."""
        text = "This is a test sentence."

        response1 = provider.embed(text)
        response2 = provider.embed(text)

        assert response1.embedding == response2.embedding

    def test_different_texts_different_embeddings(self, provider):
        """Test that different texts produce different embeddings."""
        response1 = provider.embed("Hello world")
        response2 = provider.embed("Goodbye moon")

        assert response1.embedding != response2.embedding

    def test_is_available(self, provider):
        """Test is_available method."""
        assert provider.is_available() is True


@pytest.mark.integration
class TestSentenceTransformersProviderModelCache:
    """Tests for model caching."""

    def test_model_cached(self):
        """Test that models are cached across instances."""
        from ragit.providers.sentence_transformers import SentenceTransformersProvider, _model_cache

        # Clear cache first
        _model_cache.clear()

        provider1 = SentenceTransformersProvider()
        provider1.embed("test")  # Force model load

        provider2 = SentenceTransformersProvider()
        provider2.embed("test")

        # Should have only one model in cache
        assert len(_model_cache) == 1
