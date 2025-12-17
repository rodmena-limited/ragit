#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ollama provider for LLM and Embedding operations.

This provider connects to a local or remote Ollama server.
Configuration is loaded from environment variables.
"""

import requests

from ragit.config import config
from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)


class OllamaProvider(BaseLLMProvider, BaseEmbeddingProvider):
    """
    Ollama provider for both LLM and Embedding operations.

    Parameters
    ----------
    base_url : str, optional
        Ollama server URL (default: from OLLAMA_BASE_URL env var)
    api_key : str, optional
        API key for authentication (default: from OLLAMA_API_KEY env var)
    timeout : int, optional
        Request timeout in seconds (default: from OLLAMA_TIMEOUT env var)

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
        "qwen3-embedding": 4096,
        "qwen3-embedding:0.6b": 1024,
        "qwen3-embedding:4b": 2560,
        "qwen3-embedding:8b": 4096,
    }

    # Max characters per embedding request (safe limit for 512 token models)
    MAX_EMBED_CHARS = 1500

    def __init__(
        self,
        base_url: str | None = None,
        embedding_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.embedding_url = (embedding_url or config.OLLAMA_EMBEDDING_URL).rstrip("/")
        self.api_key = api_key or config.OLLAMA_API_KEY
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self._current_embed_model: str | None = None
        self._current_dimensions: int = 768  # default

    def _get_headers(self, include_auth: bool = True) -> dict[str, str]:
        """Get request headers including authentication if API key is set."""
        headers = {"Content-Type": "application/json"}
        if include_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def dimensions(self) -> int:
        return self._current_dimensions

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._get_headers(),
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, str]]:
        """List available models on the Ollama server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._get_headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return list(data.get("models", []))
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to list Ollama models: {e}") from e

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text using Ollama."""
        options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: dict[str, str | bool | dict[str, float | int]] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self._get_headers(),
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
        """Generate embedding using Ollama (uses embedding_url, no auth for local)."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        # Truncate oversized inputs to prevent context length errors
        if len(text) > self.MAX_EMBED_CHARS:
            text = text[: self.MAX_EMBED_CHARS]

        try:
            response = requests.post(
                f"{self.embedding_url}/api/embeddings",
                headers=self._get_headers(include_auth=False),
                json={"model": model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", [])
            if not embedding:
                raise ValueError("Empty embedding returned from Ollama")

            # Update dimensions from actual response
            self._current_dimensions = len(embedding)

            return EmbeddingResponse(
                embedding=tuple(embedding),
                model=model,
                provider=self.provider_name,
                dimensions=len(embedding),
            )
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama embed failed: {e}") from e

    def embed_batch(self, texts: list[str], model: str) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts (uses embedding_url, no auth for local).

        Note: Ollama /api/embeddings only supports single prompts, so we loop.
        """
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        results = []
        try:
            for text in texts:
                # Truncate oversized inputs to prevent context length errors
                if len(text) > self.MAX_EMBED_CHARS:
                    text = text[: self.MAX_EMBED_CHARS]

                response = requests.post(
                    f"{self.embedding_url}/api/embeddings",
                    headers=self._get_headers(include_auth=False),
                    json={"model": model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding", [])
                if embedding:
                    self._current_dimensions = len(embedding)

                results.append(
                    EmbeddingResponse(
                        embedding=tuple(embedding),
                        model=model,
                        provider=self.provider_name,
                        dimensions=len(embedding),
                    )
                )
            return results
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama batch embed failed: {e}") from e

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
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
        options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: dict[str, str | bool | list[dict[str, str]] | dict[str, float | int]] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers=self._get_headers(),
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
