#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit Providers - LLM and Embedding providers for RAG optimization.

Supported providers:
- Ollama (local)
- Future: Gemini, Claude, OpenAI
"""

from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from ragit.providers.ollama import OllamaProvider

__all__ = [
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "OllamaProvider",
]
