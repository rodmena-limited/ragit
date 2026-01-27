#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit Providers - LLM and Embedding providers for RAG optimization.

Supported providers:
- OllamaProvider: Connect to local or remote Ollama servers (supports nomic-embed-text)
- FunctionProvider: Wrap custom embedding/LLM functions

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
