#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.core.experiment.experiment module.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ragit.core.experiment.experiment import (
    BenchmarkQuestion,
    Chunk,
    Document,
    EvaluationScores,
    RAGConfig,
    RagitExperiment,
    SimpleVectorStore,
)


class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_creation(self):
        """Test creating a RAGConfig."""
        config = RAGConfig(
            name="Test", chunk_size=256, chunk_overlap=50, num_chunks=3, embedding_model="embed", llm_model="llm"
        )

        assert config.name == "Test"
        assert config.chunk_size == 256
        assert config.chunk_overlap == 50
        assert config.num_chunks == 3


class TestDocument:
    """Tests for Document dataclass."""

    def test_creation(self):
        """Test creating a Document."""
        doc = Document(id="doc1", content="Hello world")

        assert doc.id == "doc1"
        assert doc.content == "Hello world"
        assert doc.metadata == {}

    def test_with_metadata(self):
        """Test Document with metadata."""
        doc = Document(id="doc1", content="Content", metadata={"source": "test.txt", "author": "Test"})

        assert doc.metadata["source"] == "test.txt"


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_creation(self):
        """Test creating a Chunk."""
        chunk = Chunk(content="chunk text", doc_id="doc1", chunk_index=0)

        assert chunk.content == "chunk text"
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_index == 0
        assert chunk.embedding is None

    def test_with_embedding(self):
        """Test Chunk with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(content="text", doc_id="doc1", chunk_index=0, embedding=embedding)

        assert chunk.embedding == embedding


class TestBenchmarkQuestion:
    """Tests for BenchmarkQuestion dataclass."""

    def test_creation(self):
        """Test creating a BenchmarkQuestion."""
        qa = BenchmarkQuestion(question="What is AI?", ground_truth="Artificial intelligence")

        assert qa.question == "What is AI?"
        assert qa.ground_truth == "Artificial intelligence"
        assert qa.relevant_doc_ids == []

    def test_with_relevant_docs(self):
        """Test BenchmarkQuestion with relevant docs."""
        qa = BenchmarkQuestion(question="Q", ground_truth="A", relevant_doc_ids=["doc1", "doc2"])

        assert qa.relevant_doc_ids == ["doc1", "doc2"]


class TestEvaluationScores:
    """Tests for EvaluationScores dataclass."""

    def test_creation(self):
        """Test creating EvaluationScores."""
        scores = EvaluationScores(answer_correctness=0.8, context_relevance=0.7, faithfulness=0.9)

        assert scores.answer_correctness == 0.8
        assert scores.context_relevance == 0.7
        assert scores.faithfulness == 0.9

    def test_combined_score(self):
        """Test combined score calculation."""
        scores = EvaluationScores(answer_correctness=1.0, context_relevance=1.0, faithfulness=1.0)

        # 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert scores.combined_score == 1.0

    def test_combined_score_weighted(self):
        """Test combined score with different values."""
        scores = EvaluationScores(answer_correctness=0.5, context_relevance=0.5, faithfulness=0.5)

        # 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
        assert scores.combined_score == 0.5


class TestSimpleVectorStore:
    """Tests for SimpleVectorStore."""

    def test_empty_store(self):
        """Test empty vector store."""
        store = SimpleVectorStore()

        assert store.chunks == []
        results = store.search([0.1, 0.2], top_k=5)
        assert results == []

    def test_add_chunks(self):
        """Test adding chunks."""
        store = SimpleVectorStore()
        chunks = [
            Chunk(content="a", doc_id="d1", chunk_index=0),
            Chunk(content="b", doc_id="d1", chunk_index=1),
        ]

        store.add(chunks)

        assert len(store.chunks) == 2

    def test_clear(self):
        """Test clearing store."""
        store = SimpleVectorStore()
        store.add([Chunk(content="a", doc_id="d1", chunk_index=0)])

        store.clear()

        assert len(store.chunks) == 0

    def test_search(self):
        """Test similarity search."""
        store = SimpleVectorStore()

        # Create chunks with embeddings
        emb1 = [1.0, 0.0, 0.0]  # Points in x direction
        emb2 = [0.0, 1.0, 0.0]  # Points in y direction
        emb3 = [0.7, 0.7, 0.0]  # Points between x and y

        chunks = [
            Chunk(content="x_chunk", doc_id="d1", chunk_index=0, embedding=emb1),
            Chunk(content="y_chunk", doc_id="d1", chunk_index=1, embedding=emb2),
            Chunk(content="xy_chunk", doc_id="d1", chunk_index=2, embedding=emb3),
        ]
        store.add(chunks)

        # Query similar to x direction
        query = [1.0, 0.0, 0.0]
        results = store.search(query, top_k=2)

        assert len(results) == 2
        # First result should be the x_chunk (exact match)
        assert results[0][0].content == "x_chunk"
        assert results[0][1] == pytest.approx(1.0, rel=0.01)

    def test_search_top_k(self):
        """Test top_k parameter."""
        store = SimpleVectorStore()

        for i in range(10):
            emb = np.random.randn(3).tolist()
            store.add([Chunk(content=f"c{i}", doc_id="d", chunk_index=i, embedding=emb)])

        query = np.random.randn(3).tolist()
        results = store.search(query, top_k=3)

        assert len(results) == 3

    def test_search_ignores_none_embeddings(self):
        """Test that chunks without embeddings are ignored."""
        store = SimpleVectorStore()

        chunks = [
            Chunk(content="with_emb", doc_id="d", chunk_index=0, embedding=[1.0, 0.0]),
            Chunk(content="no_emb", doc_id="d", chunk_index=1, embedding=None),
        ]
        store.add(chunks)

        results = store.search([1.0, 0.0], top_k=5)

        assert len(results) == 1
        assert results[0][0].content == "with_emb"


def make_mock_embed_fn():
    """Create a mock embed function."""

    def embed_fn(text: str) -> list[float]:
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        emb = np.random.randn(1024)
        emb = emb / np.linalg.norm(emb)
        return emb.tolist()

    return embed_fn


def make_mock_generate_fn(score_response: str = "85"):
    """Create a mock generate function."""

    def generate_fn(prompt: str, system_prompt: str | None = None) -> str:
        if "Rate" in prompt or "correct" in prompt.lower():
            return score_response
        return "Generated answer based on context."

    return generate_fn


class TestRagitExperiment:
    """Tests for RagitExperiment class."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embed function."""
        return make_mock_embed_fn()

    @pytest.fixture
    def mock_generate_fn(self):
        """Create a mock generate function."""
        return make_mock_generate_fn()

    @pytest.fixture
    def simple_documents(self):
        """Create simple test documents."""
        return [
            Document(id="doc1", content="Python is a programming language. It is easy to learn."),
            Document(id="doc2", content="Machine learning uses data to train models."),
        ]

    @pytest.fixture
    def simple_benchmark(self):
        """Create simple benchmark questions."""
        return [
            BenchmarkQuestion(question="What is Python?", ground_truth="Python is a programming language."),
        ]

    def test_init(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test RagitExperiment initialization."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        assert experiment.documents == simple_documents
        assert experiment.benchmark == simple_benchmark

    def test_init_requires_embed(self, simple_documents, simple_benchmark, mock_generate_fn):
        """Test that RagitExperiment requires embed_fn or provider."""
        with pytest.raises(ValueError, match="Must provide embed_fn or provider"):
            RagitExperiment(
                documents=simple_documents,
                benchmark=simple_benchmark,
                generate_fn=mock_generate_fn,
            )

    def test_init_requires_llm(self, simple_documents, simple_benchmark, mock_embed_fn):
        """Test that RagitExperiment requires LLM for evaluation."""
        with pytest.raises(ValueError, match="RagitExperiment requires LLM"):
            RagitExperiment(
                documents=simple_documents,
                benchmark=simple_benchmark,
                embed_fn=mock_embed_fn,
            )

    def test_define_search_space_defaults(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test default search space generation."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        configs = experiment.define_search_space()

        assert len(configs) > 0
        assert all(isinstance(c, RAGConfig) for c in configs)
        # All should have valid overlap < chunk_size
        assert all(c.chunk_overlap < c.chunk_size for c in configs)

    def test_define_search_space_custom(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test custom search space generation."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        configs = experiment.define_search_space(
            chunk_sizes=[100, 200],
            chunk_overlaps=[20],
            num_chunks_options=[2],
            embedding_models=["embed"],
            llm_models=["llm"],
        )

        assert len(configs) == 2  # 2 chunk sizes * 1 overlap * 1 num_chunks * 1 embed * 1 llm
        assert configs[0].chunk_size == 100
        assert configs[1].chunk_size == 200

    def test_define_search_space_filters_invalid(
        self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn
    ):
        """Test that invalid configs (overlap >= chunk_size) are filtered."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        configs = experiment.define_search_space(
            chunk_sizes=[50],
            chunk_overlaps=[50, 100],  # Both >= chunk_size
            num_chunks_options=[2],
            embedding_models=["e"],
            llm_models=["l"],
        )

        assert len(configs) == 0

    def test_chunk_document(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test document chunking."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        doc = Document(id="test", content="A" * 100)
        chunks = experiment._chunk_document(doc, chunk_size=30, overlap=10)

        assert len(chunks) > 0
        assert all(c.doc_id == "test" for c in chunks)
        assert all(len(c.content) <= 30 for c in chunks)

    def test_chunk_document_overlap(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test that chunks overlap correctly."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        doc = Document(id="test", content="0123456789" * 10)  # 100 chars
        chunks = experiment._chunk_document(doc, chunk_size=20, overlap=5)

        # Check overlapping content
        if len(chunks) > 1:
            # End of first chunk should appear at start of second
            assert chunks[1].content[:5] in chunks[0].content[-15:]

    def test_build_index(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test index building."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        config = RAGConfig(
            name="Test", chunk_size=50, chunk_overlap=10, num_chunks=2, embedding_model="embed", llm_model="llm"
        )

        experiment._build_index(config)

        assert len(experiment.vector_store.chunks) > 0
        # All chunks should have embeddings
        assert all(c.embedding is not None for c in experiment.vector_store.chunks)

    def test_retrieve(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test retrieval."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        config = RAGConfig(
            name="Test", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
        )

        experiment._build_index(config)
        chunks = experiment._retrieve("What is Python?", config)

        assert len(chunks) <= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_generate(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test generation."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        config = RAGConfig(
            name="Test", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
        )

        answer = experiment._generate(
            question="What is Python?", context="Python is a programming language.", config=config
        )

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_evaluate_response(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test response evaluation."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        config = RAGConfig(
            name="Test", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
        )

        experiment._build_index(config)
        scores = experiment._evaluate_response(
            question="What is Python?",
            generated="Python is a programming language.",
            ground_truth="Python is a programming language.",
            context="Some context",
            config=config,
        )

        assert isinstance(scores, EvaluationScores)
        assert 0 <= scores.answer_correctness <= 1
        assert 0 <= scores.context_relevance <= 1
        assert 0 <= scores.faithfulness <= 1

    def test_evaluate_config(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test evaluating a configuration."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        config = RAGConfig(
            name="TestConfig", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
        )

        result = experiment.evaluate_config(config, verbose=False)

        assert result.pattern_name == "TestConfig"
        assert result.execution_time > 0
        assert 0 <= result.final_score <= 1

    def test_run(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test running the experiment."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        configs = [
            RAGConfig(
                name="Config1", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
            ),
            RAGConfig(
                name="Config2", chunk_size=50, chunk_overlap=10, num_chunks=1, embedding_model="embed", llm_model="llm"
            ),
        ]

        results = experiment.run(configs=configs, verbose=False)

        assert len(results) == 2
        # Results should be sorted by score (best first)
        assert results[0].final_score >= results[1].final_score

    def test_run_with_max_configs(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test running with max_configs limit."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        configs = experiment.define_search_space()
        results = experiment.run(configs=configs, max_configs=1, verbose=False)

        assert len(results) == 1

    def test_get_best_config(self, simple_documents, simple_benchmark, mock_embed_fn, mock_generate_fn):
        """Test getting best configuration."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            embed_fn=mock_embed_fn,
            generate_fn=mock_generate_fn,
        )

        # Before running, should return None
        assert experiment.get_best_config() is None

        configs = [
            RAGConfig(
                name="Config1", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
            ),
        ]

        experiment.run(configs=configs, verbose=False)
        best = experiment.get_best_config()

        assert best is not None
        assert best.pattern_name == "Config1"


class MockExperimentProvider:
    """Mock provider for testing that implements both embedding and LLM interfaces."""

    def __init__(self):
        self.provider_name = "mock"
        self._dimensions = 1024

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def is_available(self) -> bool:
        return True

    def embed(self, text, model=""):
        hash_val = hash(text) % 1000
        np.random.seed(hash_val)
        emb = np.random.randn(1024)
        emb = emb / np.linalg.norm(emb)

        response = MagicMock()
        response.embedding = emb.tolist()
        return response

    def embed_batch(self, texts, model=""):
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
        response = MagicMock()
        if "Rate" in prompt or "correct" in prompt.lower():
            response.text = "85"
        else:
            response.text = "Generated answer based on context."
        return response


# Register MockExperimentProvider as implementing the base classes
from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider

BaseEmbeddingProvider.register(MockExperimentProvider)
BaseLLMProvider.register(MockExperimentProvider)


class TestRagitExperimentWithProvider:
    """Tests for RagitExperiment using provider parameter."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider that implements both embedding and LLM."""
        return MockExperimentProvider()

    @pytest.fixture
    def simple_documents(self):
        """Create simple test documents."""
        return [
            Document(id="doc1", content="Python is a programming language. It is easy to learn."),
            Document(id="doc2", content="Machine learning uses data to train models."),
        ]

    @pytest.fixture
    def simple_benchmark(self):
        """Create simple benchmark questions."""
        return [
            BenchmarkQuestion(question="What is Python?", ground_truth="Python is a programming language."),
        ]

    def test_init_with_provider(self, simple_documents, simple_benchmark, mock_provider):
        """Test RagitExperiment initialization with provider."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            provider=mock_provider,
        )

        assert experiment.documents == simple_documents
        assert experiment.benchmark == simple_benchmark

    def test_run_with_provider(self, simple_documents, simple_benchmark, mock_provider):
        """Test running with provider."""
        experiment = RagitExperiment(
            documents=simple_documents,
            benchmark=simple_benchmark,
            provider=mock_provider,
        )

        configs = [
            RAGConfig(
                name="Test", chunk_size=100, chunk_overlap=20, num_chunks=2, embedding_model="embed", llm_model="llm"
            ),
        ]

        results = experiment.run(configs=configs, verbose=False)

        assert len(results) == 1
        assert results[0].pattern_name == "Test"


class TestRagitExperimentVerbose:
    """Tests for verbose output of RagitExperiment."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embed function."""
        return make_mock_embed_fn()

    @pytest.fixture
    def mock_generate_fn(self):
        """Create a mock generate function."""
        return make_mock_generate_fn("75")

    def test_evaluate_config_verbose(self, mock_embed_fn, mock_generate_fn, capsys):
        """Test verbose output during evaluation."""
        docs = [Document(id="d1", content="Test content for verbose output testing.")]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(
            documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )
        config = RAGConfig(
            name="VerboseTest", chunk_size=100, chunk_overlap=10, num_chunks=1, embedding_model="embed", llm_model="llm"
        )

        experiment.evaluate_config(config, verbose=True)

        captured = capsys.readouterr()
        assert "VerboseTest" in captured.out
        assert "chunk_size=" in captured.out

    def test_run_verbose(self, mock_embed_fn, mock_generate_fn, capsys):
        """Test verbose output during run."""
        docs = [Document(id="d1", content="Content.")]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(
            documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )
        configs = [
            RAGConfig(name="Test", chunk_size=50, chunk_overlap=10, num_chunks=1, embedding_model="e", llm_model="l")
        ]

        experiment.run(configs=configs, verbose=True)

        captured = capsys.readouterr()
        assert "RAGIT" in captured.out
        assert "RESULTS" in captured.out

    def test_run_default_configs(self, mock_embed_fn, mock_generate_fn):
        """Test run with default search space."""
        docs = [Document(id="d1", content="A" * 1000)]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(
            documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )

        # Run with default configs but limit to 1
        results = experiment.run(max_configs=1, verbose=False)

        assert len(results) == 1


class TestScoreExtraction:
    """Tests for score extraction edge cases."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embed function."""

        def embed_fn(text: str) -> list[float]:
            return [0.1] * 100

        return embed_fn

    def test_score_extraction_no_number(self, mock_embed_fn):
        """Test score extraction when no number in response."""
        generate_fn = make_mock_generate_fn("no numbers here")

        docs = [Document(id="d1", content="Content")]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=generate_fn)
        config = RAGConfig(name="T", chunk_size=50, chunk_overlap=10, num_chunks=1, embedding_model="e", llm_model="l")

        experiment._build_index(config)
        scores = experiment._evaluate_response("Q", "A", "A", "ctx", config)

        # Should default to 0.5 when no number found
        assert 0 <= scores.answer_correctness <= 1

    def test_score_extraction_decimal(self, mock_embed_fn):
        """Test score extraction with decimal number."""
        generate_fn = make_mock_generate_fn("Score: 85.5 out of 100")

        docs = [Document(id="d1", content="Content")]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=generate_fn)
        config = RAGConfig(name="T", chunk_size=50, chunk_overlap=10, num_chunks=1, embedding_model="e", llm_model="l")

        experiment._build_index(config)
        scores = experiment._evaluate_response("Q", "A", "A", "ctx", config)

        assert 0 <= scores.answer_correctness <= 1

    def test_score_extraction_over_100(self, mock_embed_fn):
        """Test score extraction with value over 100."""
        generate_fn = make_mock_generate_fn("150")  # Over 100, should be capped

        docs = [Document(id="d1", content="Content")]
        bench = [BenchmarkQuestion(question="Q?", ground_truth="A")]

        experiment = RagitExperiment(documents=docs, benchmark=bench, embed_fn=mock_embed_fn, generate_fn=generate_fn)
        config = RAGConfig(name="T", chunk_size=50, chunk_overlap=10, num_chunks=1, embedding_model="e", llm_model="l")

        experiment._build_index(config)
        scores = experiment._evaluate_response("Q", "A", "A", "ctx", config)

        # Should be capped at 1.0
        assert scores.answer_correctness <= 1.0


class TestChunkingEdgeCases:
    """Tests for document chunking edge cases."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a mock embed function."""
        return make_mock_embed_fn()

    @pytest.fixture
    def mock_generate_fn(self):
        """Create a mock generate function."""
        return make_mock_generate_fn()

    def test_chunk_empty_document(self, mock_embed_fn, mock_generate_fn):
        """Test chunking an empty document."""
        experiment = RagitExperiment(
            documents=[], benchmark=[], embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )

        doc = Document(id="empty", content="")
        chunks = experiment._chunk_document(doc, chunk_size=100, overlap=10)

        assert len(chunks) == 0

    def test_chunk_whitespace_only(self, mock_embed_fn, mock_generate_fn):
        """Test chunking whitespace-only document."""
        experiment = RagitExperiment(
            documents=[], benchmark=[], embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )

        doc = Document(id="ws", content="   \n\t  ")
        chunks = experiment._chunk_document(doc, chunk_size=100, overlap=10)

        assert len(chunks) == 0

    def test_chunk_small_document(self, mock_embed_fn, mock_generate_fn):
        """Test chunking document smaller than chunk size."""
        experiment = RagitExperiment(
            documents=[], benchmark=[], embed_fn=mock_embed_fn, generate_fn=mock_generate_fn
        )

        doc = Document(id="small", content="Hello")
        chunks = experiment._chunk_document(doc, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello"
