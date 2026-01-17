#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit Providers - LLM and Embedding providers for RAG optimization.

Supported providers:
- OllamaProvider: Connect to local or remote Ollama servers
- FunctionProvider: Wrap custom embedding/LLM functions
- SentenceTransformersProvider: Offline embedding (requires ragit[transformers])

Base classes for implementing custom providers:
- BaseLLMProvider: Abstract base for LLM providers
- BaseEmbeddingProvider: Abstract base for embedding providers
"""

from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)
from ragit.providers.function_adapter import FunctionProvider
from ragit.providers.ollama import OllamaProvider

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "LLMResponse",
    "EmbeddingResponse",
    # Built-in providers
    "OllamaProvider",
    "FunctionProvider",
]

# Conditionally export SentenceTransformersProvider if available
try:
    from ragit.providers.sentence_transformers import (
        SentenceTransformersProvider as SentenceTransformersProvider,
    )

    __all__ += ["SentenceTransformersProvider"]
except ImportError:
    # sentence-transformers not installed, SentenceTransformersProvider not available
    pass
