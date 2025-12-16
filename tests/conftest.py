#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Shared pytest fixtures and mock helpers for ragit tests.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# Sample embedding dimensions for different models
EMBEDDING_DIMS = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "qwen3-embedding:8b": 4096,
}


def make_mock_embedding(dim: int = 1024) -> list[float]:
    """Create a normalized random embedding vector."""
    vec = np.random.randn(dim)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_URL", "http://test-embedding:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-api-key")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")
    monkeypatch.setenv("RAGIT_DEFAULT_LLM_MODEL", "test-llm")
    monkeypatch.setenv("RAGIT_DEFAULT_EMBEDDING_MODEL", "test-embed")
    monkeypatch.setenv("RAGIT_LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_ollama_responses():
    """Create mock response factory for Ollama API calls."""

    def _create_mock_response(status_code=200, json_data=None):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_data or {}
        mock_resp.raise_for_status.return_value = None
        if status_code >= 400:
            from requests import HTTPError
            mock_resp.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
        return mock_resp

    return _create_mock_response


@pytest.fixture
def mock_embedding_response(mock_ollama_responses):
    """Create mock embedding API response."""
    def _create(model: str = "mxbai-embed-large", dim: int = None):
        if dim is None:
            dim = EMBEDDING_DIMS.get(model, 1024)
        embedding = make_mock_embedding(dim)
        return mock_ollama_responses(200, {
            "embeddings": [embedding],
            "model": model,
        })
    return _create


@pytest.fixture
def mock_batch_embedding_response(mock_ollama_responses):
    """Create mock batch embedding API response."""
    def _create(count: int, model: str = "mxbai-embed-large", dim: int = None):
        if dim is None:
            dim = EMBEDDING_DIMS.get(model, 1024)
        embeddings = [make_mock_embedding(dim) for _ in range(count)]
        return mock_ollama_responses(200, {
            "embeddings": embeddings,
            "model": model,
        })
    return _create


@pytest.fixture
def mock_generate_response(mock_ollama_responses):
    """Create mock LLM generate API response."""
    def _create(text: str = "Test response", model: str = "test-llm"):
        return mock_ollama_responses(200, {
            "response": text,
            "model": model,
            "prompt_eval_count": 10,
            "eval_count": 20,
            "total_duration": 1000000000,
        })
    return _create


@pytest.fixture
def mock_chat_response(mock_ollama_responses):
    """Create mock LLM chat API response."""
    def _create(text: str = "Test chat response", model: str = "test-llm"):
        return mock_ollama_responses(200, {
            "message": {"content": text, "role": "assistant"},
            "model": model,
            "prompt_eval_count": 10,
            "eval_count": 20,
        })
    return _create


@pytest.fixture
def mock_tags_response(mock_ollama_responses):
    """Create mock models list API response."""
    return mock_ollama_responses(200, {
        "models": [
            {"name": "llama3", "size": 1000000},
            {"name": "mxbai-embed-large", "size": 500000},
        ]
    })


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from ragit import Document
    return [
        Document(
            id="doc1",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
                   "Deep learning uses neural networks with many layers. Supervised learning requires labeled data.",
            metadata={"source": "ml_intro.txt"}
        ),
        Document(
            id="doc2",
            content="Python is a programming language known for its simplicity. "
                   "It supports multiple paradigms including procedural, object-oriented, and functional programming.",
            metadata={"source": "python_intro.txt"}
        ),
        Document(
            id="doc3",
            content="RAG combines retrieval with generation for more accurate AI responses. "
                   "It retrieves relevant documents and uses them as context for the language model.",
            metadata={"source": "rag_overview.txt"}
        ),
    ]


@pytest.fixture
def sample_benchmark():
    """Sample benchmark questions for testing."""
    from ragit import BenchmarkQuestion
    return [
        BenchmarkQuestion(
            question="What is machine learning?",
            ground_truth="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            relevant_doc_ids=["doc1"]
        ),
        BenchmarkQuestion(
            question="What is RAG?",
            ground_truth="RAG combines retrieval with generation for more accurate AI responses.",
            relevant_doc_ids=["doc3"]
        ),
    ]


@pytest.fixture
def sample_rag_config():
    """Sample RAG configuration for testing."""
    from ragit import RAGConfig
    return RAGConfig(
        name="TestPattern",
        chunk_size=256,
        chunk_overlap=50,
        num_chunks=3,
        embedding_model="mxbai-embed-large",
        llm_model="test-llm",
    )


@pytest.fixture
def mock_ollama_provider(mock_embedding_response, mock_generate_response, mock_tags_response):
    """Create a fully mocked OllamaProvider."""
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get:

        def post_side_effect(url, **kwargs):
            if "/api/embed" in url:
                return mock_embedding_response()
            elif "/api/generate" in url:
                return mock_generate_response()
            elif "/api/chat" in url:
                from unittest.mock import MagicMock
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "message": {"content": "Test chat", "role": "assistant"},
                    "model": "test-llm",
                }
                resp.raise_for_status.return_value = None
                return resp
            return mock_generate_response()

        mock_post.side_effect = post_side_effect
        mock_get.return_value = mock_tags_response

        from ragit import OllamaProvider
        provider = OllamaProvider(
            base_url="http://test:11434",
            embedding_url="http://test:11434",
            api_key="test-key",
            timeout=30,
        )

        yield provider, mock_post, mock_get
