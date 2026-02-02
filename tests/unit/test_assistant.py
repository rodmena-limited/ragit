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
from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider


class MockProvider(BaseEmbeddingProvider, BaseLLMProvider):
    """Mock provider for testing that implements both base classes."""

    def __init__(self):
        self._dimensions = 1024
        # Track method calls
        self.embed_called = False
        self.generate_called = False

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def is_available(self) -> bool:
        return True

    def embed(self, text, model=""):
        self.embed_called = True
        # Create deterministic embeddings based on text hash
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        emb = np.random.randn(1024)
        emb = emb / np.linalg.norm(emb)
        response = MagicMock()
        response.embedding = emb.tolist()
        return response

    def embed_batch(self, texts, model=""):
        # Create deterministic embeddings for batch
        responses = []
        for text in texts:
            hash_val = hash(text) % 1000
            np.random.seed(hash_val)
            emb = np.random.randn(1024)
            emb = emb / np.linalg.norm(emb)
            response = MagicMock()
            response.embedding = emb.tolist()
            responses.append(response)
        return responses

    def generate(self, prompt, model="", system_prompt=None, temperature=0.7, max_tokens=None):
        self.generate_called = True
        response = MagicMock()
        response.text = "Generated response based on context."
        return response


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    return MockProvider()


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

    def test_init_requires_provider_or_embed_fn(self, sample_documents):
        """Test that initialization requires embed_fn or provider."""
        with pytest.raises(ValueError, match="Must provide embed_fn or provider"):
            RAGAssistant(sample_documents)

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
        assert mock_provider.generate_called


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
        assert mock_provider.embed_called
        # Provider.generate should be called
        assert mock_provider.generate_called

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

        class MarkdownMockProvider(BaseEmbeddingProvider, BaseLLMProvider):
            """Mock provider that returns markdown code."""

            @property
            def provider_name(self) -> str:
                return "mock"

            @property
            def dimensions(self) -> int:
                return 1024

            def is_available(self) -> bool:
                return True

            def embed(self, text, model=""):
                response = MagicMock()
                response.embedding = [0.1] * 1024
                return response

            def embed_batch(self, texts, model=""):
                responses = []
                for _ in texts:
                    response = MagicMock()
                    response.embedding = [0.1] * 1024
                    responses.append(response)
                return responses

            def generate(self, prompt, model="", system_prompt=None, temperature=0.7, max_tokens=None):
                response = MagicMock()
                response.text = "```python\nprint('hello')\n```"
                return response

        assistant = RAGAssistant(sample_documents, provider=MarkdownMockProvider())
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

    def test_has_llm_with_full_provider(self, sample_documents, mock_provider):
        """Test has_llm property with provider that has LLM."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        assert assistant.has_llm is True

    def test_has_llm_with_embed_only(self, sample_documents):
        """Test has_llm property with embed_fn only (no LLM)."""

        def mock_embed(text):
            return [0.1] * 1024

        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed)
        assert assistant.has_llm is False


class TestRAGAssistantEmbedFn:
    """Tests for RAGAssistant with embed_fn parameter."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embedding function."""

        def embed_fn(text: str) -> list[float]:
            hash_val = hash(text) % 1000
            np.random.seed(hash_val)
            emb = np.random.randn(1024)
            emb = emb / np.linalg.norm(emb)
            return emb.tolist()

        return embed_fn

    @pytest.fixture
    def mock_generate_fn(self):
        """Create a mock generation function."""

        def generate_fn(prompt: str, system_prompt: str | None = None) -> str:
            return "Generated response."

        return generate_fn

    def test_init_with_embed_fn(self, sample_documents, mock_embed_fn):
        """Test initialization with embed_fn."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        assert assistant.num_documents == 3
        assert assistant.num_chunks > 0
        assert assistant.has_llm is False

    def test_init_with_embed_and_generate_fn(self, sample_documents, mock_embed_fn, mock_generate_fn):
        """Test initialization with both embed_fn and generate_fn."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn, generate_fn=mock_generate_fn)

        assert assistant.has_llm is True

    def test_retrieve_works_without_llm(self, sample_documents, mock_embed_fn):
        """Test that retrieve works without LLM configured."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        results = assistant.retrieve("Python", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)

    def test_get_context_works_without_llm(self, sample_documents, mock_embed_fn):
        """Test that get_context works without LLM configured."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        context = assistant.get_context("Python", top_k=2)

        assert isinstance(context, str)
        assert len(context) > 0

    def test_ask_raises_without_llm(self, sample_documents, mock_embed_fn):
        """Test that ask raises NotImplementedError without LLM."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        with pytest.raises(NotImplementedError, match="No LLM configured"):
            assistant.ask("What is Python?")

    def test_generate_raises_without_llm(self, sample_documents, mock_embed_fn):
        """Test that generate raises NotImplementedError without LLM."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        with pytest.raises(NotImplementedError, match="No LLM configured"):
            assistant.generate("Tell me about Python")

    def test_generate_code_raises_without_llm(self, sample_documents, mock_embed_fn):
        """Test that generate_code raises NotImplementedError without LLM."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn)

        with pytest.raises(NotImplementedError, match="No LLM configured"):
            assistant.generate_code("hello world function")

    def test_ask_works_with_generate_fn(self, sample_documents, mock_embed_fn, mock_generate_fn):
        """Test that ask works with generate_fn."""
        assistant = RAGAssistant(sample_documents, embed_fn=mock_embed_fn, generate_fn=mock_generate_fn)

        answer = assistant.ask("What is Python?")

        assert isinstance(answer, str)
        assert len(answer) > 0


class TestRAGAssistantThreadSafety:
    """Tests for thread safety in RAGAssistant."""

    def test_concurrent_add_and_retrieve(self, sample_documents, mock_provider):
        """Verify lock-free design handles concurrent access."""
        import threading

        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        errors = []

        def reader():
            for _ in range(50):
                try:
                    results = assistant.retrieve("test query", top_k=3)
                    # Verify we get consistent results (not partial state)
                    assert isinstance(results, list)
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(5):
                try:
                    assistant.add_documents([Document(id=f"new-{i}", content=f"new content number {i} for testing")])
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        threads += [threading.Thread(target=writer) for _ in range(2)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_retrieve_only(self, sample_documents, mock_provider):
        """Verify concurrent reads don't interfere with each other."""
        import threading

        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        results_list = []
        errors = []

        def reader(query):
            try:
                results = assistant.retrieve(query, top_k=2)
                results_list.append(results)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(f"query {i}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results_list) == 10


class TestRAGAssistantPersistence:
    """Tests for index persistence (save/load)."""

    def test_save_load_index(self, tmp_path, sample_documents, mock_provider):
        """Verify index survives save/load cycle."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        original_results = assistant.retrieve("Python programming", top_k=3)

        # Save
        index_path = tmp_path / "index"
        assistant.save_index(index_path)

        # Verify files exist
        assert (index_path / "chunks.json").exists()
        assert (index_path / "embeddings.npy").exists()
        assert (index_path / "metadata.json").exists()

        # Load
        loaded = RAGAssistant.load_index(index_path, provider=mock_provider)

        # Verify same chunk count
        assert loaded.num_chunks == assistant.num_chunks

        # Verify same retrieval results
        loaded_results = loaded.retrieve("Python programming", top_k=3)

        assert len(original_results) == len(loaded_results)
        for (orig_chunk, orig_score), (loaded_chunk, loaded_score) in zip(
            original_results, loaded_results, strict=True
        ):
            assert orig_chunk.doc_id == loaded_chunk.doc_id
            assert orig_chunk.content == loaded_chunk.content
            assert abs(orig_score - loaded_score) < 1e-6

    def test_save_creates_directory(self, tmp_path, sample_documents, mock_provider):
        """Test that save_index creates the directory if it doesn't exist."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)

        nested_path = tmp_path / "a" / "b" / "c" / "index"
        assistant.save_index(nested_path)

        assert nested_path.exists()
        assert (nested_path / "chunks.json").exists()

    def test_load_validates_consistency(self, tmp_path, sample_documents, mock_provider):
        """Verify load_index detects corrupted index."""
        import json

        from ragit.exceptions import IndexingError

        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        index_path = tmp_path / "index"
        assistant.save_index(index_path)

        # Corrupt by adding extra chunk to JSON
        chunks_path = index_path / "chunks.json"
        chunks_data = json.loads(chunks_path.read_text())
        chunks_data.append({"id": "extra", "content": "extra chunk", "doc_id": "x"})
        chunks_path.write_text(json.dumps(chunks_data))

        with pytest.raises(IndexingError, match="corrupted"):
            RAGAssistant.load_index(index_path, provider=mock_provider)

    def test_load_empty_index(self, tmp_path, mock_provider):
        """Test loading an index with no chunks."""
        import json

        index_path = tmp_path / "empty_index"
        index_path.mkdir()

        # Create empty index files
        (index_path / "chunks.json").write_text("[]")
        (index_path / "metadata.json").write_text(
            json.dumps({"chunk_count": 0, "embedding_model": "test", "version": "1.0"})
        )
        # No embeddings.npy file (empty index)

        loaded = RAGAssistant.load_index(index_path, provider=mock_provider)
        assert loaded.num_chunks == 0
        assert not loaded.is_indexed

    def test_is_indexed_property(self, sample_documents, mock_provider):
        """Test is_indexed property."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        assert assistant.is_indexed is True

    def test_chunk_count_property(self, sample_documents, mock_provider):
        """Test chunk_count property."""
        assistant = RAGAssistant(sample_documents, provider=mock_provider)
        assert assistant.chunk_count == assistant.num_chunks


class TestRAGAssistantEmptyDocuments:
    """Tests for empty document handling."""

    def test_empty_documents_list(self, mock_provider):
        """Test initialization with empty document list."""
        assistant = RAGAssistant([], provider=mock_provider)

        assert assistant.num_chunks == 0
        assert not assistant.is_indexed
        assert assistant.retrieve("test") == []

    def test_documents_with_no_content(self, mock_provider):
        """Test documents with empty content produce no chunks."""
        docs = [
            Document(id="empty1", content=""),
            Document(id="empty2", content="   "),
        ]

        assistant = RAGAssistant(docs, provider=mock_provider)

        assert assistant.num_chunks == 0
        assert not assistant.is_indexed


class TestRAGAssistantEmbeddingValidation:
    """Tests for embedding count validation."""

    def test_embedding_count_mismatch_raises(self):
        """Verify mismatch between chunks and embeddings raises error."""
        from ragit.exceptions import IndexingError
        from ragit.providers.base import BaseEmbeddingProvider

        class BadProvider(BaseEmbeddingProvider):
            """Provider that returns fewer embeddings than requested."""

            @property
            def provider_name(self) -> str:
                return "bad"

            @property
            def dimensions(self) -> int:
                return 768

            def is_available(self) -> bool:
                return True

            def embed(self, text, model=""):
                response = MagicMock()
                response.embedding = [0.1] * 768
                return response

            def embed_batch(self, texts, model=""):
                # Return fewer embeddings than requested!
                responses = []
                for i, _ in enumerate(texts):
                    if i < len(texts) - 1:  # Skip last one
                        response = MagicMock()
                        response.embedding = [0.1] * 768
                        responses.append(response)
                return responses

        docs = [
            Document(id="doc1", content="Some content here"),
            Document(id="doc2", content="More content here"),
        ]

        with pytest.raises(IndexingError, match="Embedding count mismatch"):
            RAGAssistant(docs, provider=BadProvider())

    def test_add_documents_validates_embeddings(self, sample_documents):
        """Test add_documents validates embedding count."""
        from ragit.exceptions import IndexingError
        from ragit.providers.base import BaseEmbeddingProvider

        call_count = [0]

        class PartiallyBadProvider(BaseEmbeddingProvider):
            """Provider that fails on second batch."""

            @property
            def provider_name(self) -> str:
                return "partial"

            @property
            def dimensions(self) -> int:
                return 768

            def is_available(self) -> bool:
                return True

            def embed(self, text, model=""):
                response = MagicMock()
                response.embedding = [0.1] * 768
                return response

            def embed_batch(self, texts, model=""):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call (initial indexing) - return correct count
                    return [MagicMock(embedding=[0.1] * 768) for _ in texts]
                else:
                    # Second call (add_documents) - return empty (fewer than requested)
                    return []

        assistant = RAGAssistant(sample_documents, provider=PartiallyBadProvider())

        with pytest.raises(IndexingError, match="Embedding count mismatch"):
            assistant.add_documents([Document(id="new", content="new document content")])
