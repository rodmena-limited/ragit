#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Base provider interfaces for LLM and Embedding providers.

These abstract classes define the interface that all providers must implement,
making it easy to add new providers (Gemini, Claude, OpenAI, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    provider: str
    usage: dict[str, int] | None = None


@dataclass
class EmbeddingResponse:
    """Response from an embedding call."""

    embedding: list[float]
    model: str
    provider: str
    dimensions: int


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Implement this to add support for new LLM providers like Gemini, Claude, etc.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'ollama', 'gemini', 'claude')."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate text from the LLM.

        Parameters
        ----------
        prompt : str
            The user prompt/query.
        model : str
            Model identifier (e.g., 'llama3', 'qwen3-vl:235b-instruct-cloud').
        system_prompt : str, optional
            System prompt for context/instructions.
        temperature : float
            Sampling temperature (0.0 to 1.0).
        max_tokens : int, optional
            Maximum tokens to generate.

        Returns
        -------
        LLMResponse
            The generated response.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implement this to add support for new embedding providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions for the current model."""
        pass

    @abstractmethod
    def embed(self, text: str, model: str) -> EmbeddingResponse:
        """
        Generate embedding for text.

        Parameters
        ----------
        text : str
            Text to embed.
        model : str
            Model identifier (e.g., 'nomic-embed-text').

        Returns
        -------
        EmbeddingResponse
            The embedding response.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str], model: str) -> list[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        model : str
            Model identifier.

        Returns
        -------
        list[EmbeddingResponse]
            List of embedding responses.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass
