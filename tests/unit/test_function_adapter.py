#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.providers.function_adapter module.
"""

import numpy as np
import pytest

from ragit.providers.function_adapter import FunctionProvider


class TestFunctionProviderEmbedding:
    """Tests for FunctionProvider embedding functionality."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embedding function."""

        def embed_fn(text: str) -> list[float]:
            # Create deterministic embeddings based on text hash
            hash_val = hash(text) % 1000
            np.random.seed(hash_val)
            emb = np.random.randn(384)
            return (emb / np.linalg.norm(emb)).tolist()

        return embed_fn

    def test_provider_name(self, mock_embed_fn):
        """Test provider name."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)
        assert provider.provider_name == "function"

    def test_has_embedding(self, mock_embed_fn):
        """Test has_embedding property."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)
        assert provider.has_embedding is True

        provider_no_embed = FunctionProvider(generate_fn=lambda x: "test")
        assert provider_no_embed.has_embedding is False

    def test_has_llm(self, mock_embed_fn):
        """Test has_llm property."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)
        assert provider.has_llm is False

        provider_with_llm = FunctionProvider(embed_fn=mock_embed_fn, generate_fn=lambda x: "test")
        assert provider_with_llm.has_llm is True

    def test_is_available(self, mock_embed_fn):
        """Test is_available method."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)
        assert provider.is_available() is True

        empty_provider = FunctionProvider()
        assert empty_provider.is_available() is False

    def test_embed(self, mock_embed_fn):
        """Test embed method."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)

        response = provider.embed("test text")

        assert response.provider == "function"
        assert response.model == "function"
        assert len(response.embedding) == 384
        assert response.dimensions == 384

    def test_embed_returns_tuple(self, mock_embed_fn):
        """Test that embed returns tuple for immutability."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)

        response = provider.embed("test")

        assert isinstance(response.embedding, tuple)

    def test_embed_auto_detects_dimensions(self, mock_embed_fn):
        """Test that dimensions are auto-detected."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)

        # Before first embed, dimensions raises error
        with pytest.raises(ValueError, match="Embedding dimensions not yet determined"):
            _ = provider.dimensions

        # After embed, dimensions are set
        provider.embed("test")
        assert provider.dimensions == 384

    def test_embed_with_explicit_dimensions(self, mock_embed_fn):
        """Test embed with explicit dimensions."""
        provider = FunctionProvider(embed_fn=mock_embed_fn, embedding_dimensions=384)

        assert provider.dimensions == 384

    def test_embed_raises_without_embed_fn(self):
        """Test that embed raises error without embed_fn."""
        provider = FunctionProvider()

        with pytest.raises(ValueError, match="No embedding function configured"):
            provider.embed("test")

    def test_embed_batch(self, mock_embed_fn):
        """Test embed_batch method."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)

        texts = ["text1", "text2", "text3"]
        responses = provider.embed_batch(texts)

        assert len(responses) == 3
        assert all(len(r.embedding) == 384 for r in responses)

    def test_embed_consistent(self, mock_embed_fn):
        """Test that embeddings are consistent for same input."""
        provider = FunctionProvider(embed_fn=mock_embed_fn)

        response1 = provider.embed("hello world")
        response2 = provider.embed("hello world")

        assert response1.embedding == response2.embedding


class TestFunctionProviderGeneration:
    """Tests for FunctionProvider generation functionality."""

    def test_generate_single_arg(self):
        """Test generate with single-arg function."""

        def generate_fn(prompt: str) -> str:
            return f"Response to: {prompt}"

        provider = FunctionProvider(generate_fn=generate_fn)

        response = provider.generate("Hello")

        assert "Hello" in response.text
        assert response.provider == "function"

    def test_generate_two_args(self):
        """Test generate with two-arg function."""

        def generate_fn(prompt: str, system_prompt: str | None) -> str:
            if system_prompt:
                return f"System: {system_prompt}, Prompt: {prompt}"
            return f"Prompt: {prompt}"

        provider = FunctionProvider(generate_fn=generate_fn)

        response = provider.generate("Hello", system_prompt="Be helpful")

        assert "Be helpful" in response.text
        assert "Hello" in response.text

    def test_generate_single_arg_with_system_prompt(self):
        """Test that single-arg function prepends system prompt."""

        def generate_fn(prompt: str) -> str:
            return prompt

        provider = FunctionProvider(generate_fn=generate_fn)

        response = provider.generate("User question", system_prompt="System instruction")

        # System prompt should be prepended
        assert "System instruction" in response.text
        assert "User question" in response.text

    def test_generate_raises_without_generate_fn(self):
        """Test that generate raises error without generate_fn."""

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 100

        provider = FunctionProvider(embed_fn=embed_fn)

        with pytest.raises(NotImplementedError, match="No LLM configured"):
            provider.generate("test")


class TestFunctionProviderCombined:
    """Tests for FunctionProvider with both embedding and generation."""

    @pytest.fixture
    def combined_provider(self):
        """Create a provider with both embed and generate functions."""

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 768

        def generate_fn(prompt: str, system_prompt: str | None = None) -> str:
            return f"Generated: {prompt[:20]}..."

        return FunctionProvider(embed_fn=embed_fn, generate_fn=generate_fn)

    def test_combined_provider(self, combined_provider):
        """Test provider with both functions."""
        assert combined_provider.has_embedding is True
        assert combined_provider.has_llm is True
        assert combined_provider.is_available() is True

    def test_combined_embed(self, combined_provider):
        """Test embedding on combined provider."""
        response = combined_provider.embed("test")
        assert len(response.embedding) == 768

    def test_combined_generate(self, combined_provider):
        """Test generation on combined provider."""
        response = combined_provider.generate("Hello world")
        assert "Hello world" in response.text


class TestFunctionProviderSignatureDetection:
    """Tests for generate_fn signature detection."""

    def test_detects_single_required_param(self):
        """Test detection of single required parameter."""

        def fn(prompt: str) -> str:
            return prompt

        provider = FunctionProvider(generate_fn=fn)
        assert provider._generate_fn_signature == 1

    def test_detects_two_required_params(self):
        """Test detection of two required parameters."""

        def fn(prompt: str, system: str) -> str:
            return prompt + system

        provider = FunctionProvider(generate_fn=fn)
        assert provider._generate_fn_signature == 2

    def test_detects_with_optional_params(self):
        """Test detection with optional parameters."""

        def fn(prompt: str, system: str | None = None) -> str:
            return prompt

        provider = FunctionProvider(generate_fn=fn)
        # Has 1 required, but also accepts optional, so signature is 2
        assert provider._generate_fn_signature in (1, 2)

    def test_lambda_single_arg(self):
        """Test detection with lambda single arg."""
        provider = FunctionProvider(generate_fn=lambda x: x)
        assert provider._generate_fn_signature == 1
