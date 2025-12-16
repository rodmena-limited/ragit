#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.assistant module.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ragit import Document
from ragit.assistant import RAGAssistant


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.provider_name = "mock"

    def mock_embed(text, model):
        # Create deterministic embeddings based on text hash
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        emb = np.random.randn(1024)
        emb = emb / np.linalg.norm(emb)
        response = MagicMock()
        response.embedding = emb.tolist()
        return response

    provider.embed.side_effect = mock_embed

    def mock_generate(prompt, model, system_prompt=None, temperature=0.7):
        response = MagicMock()
        response.text = "Generated response based on context."
        return response

    provider.generate.side_effect = mock_generate

    return provider


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Python is a programming language. It is easy to learn and use.",
        ),
        Document(
            id="doc2",
            content="Falcon is a web framework for Python. It is fast and lightweight.",
        ),
        Document(
            id="doc3",
            content="REST APIs use HTTP methods like GET, POST, PUT, DELETE.",
        ),
    ]


class TestRAGAssistantInit:
    """Tests for RAGAssistant initialization."""

    def test_init_with_documents(self, sample_documents, mock_provider):
        """Test initialization with document list."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        assert assistant.num_documents == 3
        assert assistant.num_chunks > 0

    def test_init_with_file_path(self, tmp_path, mock_provider):
        """Test initialization with file path."""
        file_path = tmp_path / "doc.txt"
        file_path.write_text("Test content for the document.")

        assistant = RAGAssistant(str(file_path), provider=mock_provider)

        assert assistant.num_documents == 1

    def test_init_with_directory(self, tmp_path, mock_provider):
        """Test initialization with directory path."""
        (tmp_path / "a.txt").write_text("Content A")
        (tmp_path / "b.txt").write_text("Content B")

        assistant = RAGAssistant(tmp_path, provider=mock_provider)

        assert assistant.num_documents == 2

    def test_init_with_invalid_path(self, mock_provider):
        """Test initialization with invalid path raises error."""
        with pytest.raises(ValueError, match="Invalid documents source"):
            RAGAssistant("/nonexistent/path/file.txt", provider=mock_provider)

    def test_init_custom_models(self, sample_documents, mock_provider):
        """Test initialization with custom model names."""
        assistant = RAGAssistant(
            sample_documents, provider=mock_provider, embedding_model="custom-embed", llm_model="custom-llm"
        )

        assert assistant.embedding_model == "custom-embed"
        assert assistant.llm_model == "custom-llm"

    def test_init_custom_chunk_params(self, sample_documents, mock_provider):
        """Test initialization with custom chunking parameters."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider, chunk_size=100, chunk_overlap=20)

        assert assistant.chunk_size == 100
        assert assistant.chunk_overlap == 20


class TestRAGAssistantRetrieve:
    """Tests for RAGAssistant.retrieve method."""

    def test_retrieve_returns_results(self, sample_documents, mock_provider):
        """Test that retrieve returns results."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        results = assistant.retrieve("What is Python?", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_retrieve_returns_scores(self, sample_documents, mock_provider):
        """Test that retrieve returns valid similarity scores."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        results = assistant.retrieve("Python programming", top_k=3)

        for _chunk, score in results:
            assert -1 <= score <= 1  # Cosine similarity range

    def test_retrieve_top_k(self, sample_documents, mock_provider):
        """Test that top_k limits results."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        results_1 = assistant.retrieve("test", top_k=1)
        results_3 = assistant.retrieve("test", top_k=3)

        assert len(results_1) == 1
        assert len(results_3) <= 3

    def test_retrieve_sorted_by_score(self, sample_documents, mock_provider):
        """Test that results are sorted by score descending."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        results = assistant.retrieve("test query", top_k=3)
        scores = [score for _, score in results]

        assert scores == sorted(scores, reverse=True)


class TestRAGAssistantGetContext:
    """Tests for RAGAssistant.get_context method."""

    def test_get_context_returns_string(self, sample_documents, mock_provider):
        """Test that get_context returns a string."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        context = assistant.get_context("Python", top_k=2)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_get_context_contains_chunk_content(self, sample_documents, mock_provider):
        """Test that context contains retrieved chunk content."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        context = assistant.get_context("Python", top_k=3)

        # Should contain content from at least one document
        assert any(word in context for word in ["Python", "Falcon", "REST", "HTTP"])


class TestRAGAssistantGenerate:
    """Tests for RAGAssistant.generate method."""

    def test_generate_returns_string(self, sample_documents, mock_provider):
        """Test that generate returns a string."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        response = assistant.generate("Tell me about Python")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_with_system_prompt(self, sample_documents, mock_provider):
        """Test generation with custom system prompt."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        response = assistant.generate("Hello", system_prompt="You are a helpful assistant.")

        assert isinstance(response, str)

    def test_generate_calls_provider(self, sample_documents, mock_provider):
        """Test that generate calls the provider."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        assistant.generate("Test prompt")

        # Provider.generate should have been called
        assert mock_provider.generate.called


class TestRAGAssistantAsk:
    """Tests for RAGAssistant.ask method."""

    def test_ask_returns_string(self, sample_documents, mock_provider):
        """Test that ask returns a string."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        answer = assistant.ask("What is Python?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_uses_retrieval(self, sample_documents, mock_provider):
        """Test that ask retrieves context before generating."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        assistant.ask("What is Falcon?", top_k=2)

        # Provider.embed should be called for query
        assert mock_provider.embed.called
        # Provider.generate should be called
        assert mock_provider.generate.called

    def test_ask_with_custom_params(self, sample_documents, mock_provider):
        """Test ask with custom parameters."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        answer = assistant.ask("Question", system_prompt="Custom prompt", top_k=1, temperature=0.5)

        assert isinstance(answer, str)


class TestRAGAssistantGenerateCode:
    """Tests for RAGAssistant.generate_code method."""

    def test_generate_code_returns_string(self, sample_documents, mock_provider):
        """Test that generate_code returns a string."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        code = assistant.generate_code("create a hello world function")

        assert isinstance(code, str)

    def test_generate_code_cleans_markdown(self, sample_documents):
        """Test that markdown code blocks are removed."""
        mock_prov = MagicMock()

        def mock_embed(text, model):
            response = MagicMock()
            response.embedding = [0.1] * 1024
            return response

        mock_prov.embed.side_effect = mock_embed

        def mock_generate(prompt, model, system_prompt=None, temperature=0.7):
            response = MagicMock()
            response.text = "```python\nprint('hello')\n```"
            return response

        mock_prov.generate.side_effect = mock_generate

        assistant = RAGAssistant(sample_documents, provider=mock_prov)
        code = assistant.generate_code("hello world")

        assert "```" not in code
        assert "print('hello')" in code

    def test_generate_code_with_language(self, sample_documents, mock_provider):
        """Test generating code with specific language."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        code = assistant.generate_code("create a function", language="javascript")

        assert isinstance(code, str)


class TestRAGAssistantProperties:
    """Tests for RAGAssistant properties."""

    def test_num_chunks(self, sample_documents, mock_provider):
        """Test num_chunks property."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        assert assistant.num_chunks > 0
        assert isinstance(assistant.num_chunks, int)

    def test_num_documents(self, sample_documents, mock_provider):
        """Test num_documents property."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        assert assistant.num_documents == 3


class TestRAGAssistantCosine:
    """Tests for cosine similarity calculation."""

    def test_cosine_identical_vectors(self, sample_documents, mock_provider):
        """Test cosine similarity of identical vectors."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        vec = [1.0, 0.0, 0.0]
        sim = assistant._cosine_similarity(vec, vec)

        assert sim == pytest.approx(1.0, rel=0.001)

    def test_cosine_orthogonal_vectors(self, sample_documents, mock_provider):
        """Test cosine similarity of orthogonal vectors."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = assistant._cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(0.0, abs=0.001)

    def test_cosine_opposite_vectors(self, sample_documents, mock_provider):
        """Test cosine similarity of opposite vectors."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        sim = assistant._cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(-1.0, rel=0.001)
