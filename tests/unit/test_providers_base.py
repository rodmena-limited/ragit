#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.providers.base module.
"""

import pytest
from ragit.providers.base import (
    LLMResponse,
    EmbeddingResponse,
    BaseLLMProvider,
    BaseEmbeddingProvider,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_llm_response_creation(self):
        """Test creating an LLMResponse."""
        response = LLMResponse(
            text="Hello world",
            model="llama3",
            provider="ollama",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )

        assert response.text == "Hello world"
        assert response.model == "llama3"
        assert response.provider == "ollama"
        assert response.usage["prompt_tokens"] == 10

    def test_llm_response_without_usage(self):
        """Test LLMResponse with no usage info."""
        response = LLMResponse(
            text="Test",
            model="model",
            provider="provider"
        )

        assert response.usage is None

    def test_llm_response_equality(self):
        """Test LLMResponse equality."""
        r1 = LLMResponse(text="a", model="m", provider="p")
        r2 = LLMResponse(text="a", model="m", provider="p")

        assert r1 == r2


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse dataclass."""

    def test_embedding_response_creation(self):
        """Test creating an EmbeddingResponse."""
        embedding = [0.1, 0.2, 0.3]
        response = EmbeddingResponse(
            embedding=embedding,
            model="nomic-embed-text",
            provider="ollama",
            dimensions=3
        )

        assert response.embedding == embedding
        assert response.model == "nomic-embed-text"
        assert response.provider == "ollama"
        assert response.dimensions == 3

    def test_embedding_response_dimensions(self):
        """Test that dimensions matches embedding length."""
        embedding = [0.0] * 768
        response = EmbeddingResponse(
            embedding=embedding,
            model="nomic-embed-text",
            provider="ollama",
            dimensions=768
        )

        assert len(response.embedding) == response.dimensions


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_concrete_implementation(self):
        """Test implementing BaseLLMProvider."""

        class ConcreteLLMProvider(BaseLLMProvider):
            @property
            def provider_name(self) -> str:
                return "test_provider"

            def generate(self, prompt, model, system_prompt=None, temperature=0.7, max_tokens=None):
                return LLMResponse(text="test", model=model, provider=self.provider_name)

            def is_available(self) -> bool:
                return True

        provider = ConcreteLLMProvider()

        assert provider.provider_name == "test_provider"
        assert provider.is_available() is True

        response = provider.generate("Hello", "model")
        assert response.text == "test"


class TestBaseEmbeddingProvider:
    """Tests for BaseEmbeddingProvider abstract class."""

    def test_cannot_instantiate_base_class(self):
        """Test that BaseEmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbeddingProvider()

    def test_concrete_implementation(self):
        """Test implementing BaseEmbeddingProvider."""

        class ConcreteEmbeddingProvider(BaseEmbeddingProvider):
            @property
            def provider_name(self) -> str:
                return "test_embed"

            @property
            def dimensions(self) -> int:
                return 768

            def embed(self, text, model):
                return EmbeddingResponse(
                    embedding=[0.0] * 768,
                    model=model,
                    provider=self.provider_name,
                    dimensions=768
                )

            def embed_batch(self, texts, model):
                return [self.embed(t, model) for t in texts]

            def is_available(self) -> bool:
                return True

        provider = ConcreteEmbeddingProvider()

        assert provider.provider_name == "test_embed"
        assert provider.dimensions == 768
        assert provider.is_available() is True

        response = provider.embed("hello", "model")
        assert len(response.embedding) == 768

        batch = provider.embed_batch(["a", "b"], "model")
        assert len(batch) == 2
