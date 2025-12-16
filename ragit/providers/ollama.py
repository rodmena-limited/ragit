#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ollama provider for LLM and Embedding operations.

This provider connects to a local or remote Ollama server.
"""

import requests
from typing import Optional

from ragit.providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    LLMResponse,
    EmbeddingResponse,
)


class OllamaProvider(BaseLLMProvider, BaseEmbeddingProvider):
    """
    Ollama provider for both LLM and Embedding operations.

    Parameters
    ----------
    base_url : str
        Ollama server URL (default: http://localhost:11434)
    timeout : int
        Request timeout in seconds (default: 120)

    Examples
    --------
    >>> provider = OllamaProvider()
    >>> response = provider.generate("What is RAG?", model="llama3")
    >>> print(response.text)

    >>> embedding = provider.embed("Hello world", model="nomic-embed-text")
    >>> print(len(embedding.embedding))
    """

    # Known embedding model dimensions
    EMBEDDING_DIMENSIONS = {
        "nomic-embed-text": 768,
        "nomic-embed-text:latest": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._current_embed_model: Optional[str] = None
        self._current_dimensions: int = 768  # default

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimensions(self) -> int:
        return self._current_dimensions

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict]:
        """List available models on the Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to list Ollama models: {e}") from e

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text using Ollama."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                text=data.get("response", ""),
                model=model,
                provider=self.provider_name,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                    "total_duration": data.get("total_duration"),
                },
            )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama generate failed: {e}") from e

    def embed(self, text: str, model: str) -> EmbeddingResponse:
        """Generate embedding using Ollama."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embeddings", [[]])[0]
            if not embedding:
                raise ValueError("Empty embedding returned from Ollama")

            # Update dimensions from actual response
            self._current_dimensions = len(embedding)

            return EmbeddingResponse(
                embedding=embedding,
                model=model,
                provider=self.provider_name,
                dimensions=len(embedding),
            )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama embed failed: {e}") from e

    def embed_batch(self, texts: list[str], model: str) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings", [])
            if embeddings:
                self._current_dimensions = len(embeddings[0])

            return [
                EmbeddingResponse(
                    embedding=emb,
                    model=model,
                    provider=self.provider_name,
                    dimensions=len(emb),
                )
                for emb in embeddings
            ]
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama batch embed failed: {e}") from e

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Chat completion using Ollama.

        Parameters
        ----------
        messages : list[dict]
            List of messages with 'role' and 'content' keys.
        model : str
            Model identifier.
        temperature : float
            Sampling temperature.
        max_tokens : int, optional
            Maximum tokens to generate.

        Returns
        -------
        LLMResponse
            The generated response.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                text=data.get("message", {}).get("content", ""),
                model=model,
                provider=self.provider_name,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                },
            )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama chat failed: {e}") from e
