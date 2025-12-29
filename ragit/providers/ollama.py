#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ollama provider for LLM and Embedding operations.

This provider connects to a local or remote Ollama server.
Configuration is loaded from environment variables.

Performance optimizations:
- Connection pooling via requests.Session()
- Async parallel embedding via trio + httpx
- LRU cache for repeated embedding queries
"""

from functools import lru_cache
from typing import Any

import httpx
import requests

from ragit.config import config
from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)


# Module-level cache for embeddings (shared across instances)
@lru_cache(maxsize=2048)
def _cached_embedding(text: str, model: str, embedding_url: str, timeout: int) -> tuple[float, ...]:
    """Cache embedding results to avoid redundant API calls."""
    # Truncate oversized inputs
    if len(text) > OllamaProvider.MAX_EMBED_CHARS:
        text = text[: OllamaProvider.MAX_EMBED_CHARS]

    response = requests.post(
        f"{embedding_url}/api/embed",
        headers={"Content-Type": "application/json"},
        json={"model": model, "input": text},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings", [])
    if not embeddings or not embeddings[0]:
        raise ValueError("Empty embedding returned from Ollama")
    return tuple(embeddings[0])


class OllamaProvider(BaseLLMProvider, BaseEmbeddingProvider):
    """
    Ollama provider for both LLM and Embedding operations.

    Performance features:
    - Connection pooling via requests.Session() for faster sequential requests
    - Native batch embedding via /api/embed endpoint (single API call)
    - LRU cache for repeated embedding queries (2048 entries)

    Parameters
    ----------
    base_url : str, optional
        Ollama server URL (default: from OLLAMA_BASE_URL env var)
    api_key : str, optional
        API key for authentication (default: from OLLAMA_API_KEY env var)
    timeout : int, optional
        Request timeout in seconds (default: from OLLAMA_TIMEOUT env var)
    use_cache : bool, optional
        Enable embedding cache (default: True)

    Examples
    --------
    >>> provider = OllamaProvider()
    >>> response = provider.generate("What is RAG?", model="llama3")
    >>> print(response.text)

    >>> # Batch embedding (single API call)
    >>> embeddings = provider.embed_batch(texts, "mxbai-embed-large")
    """

    # Known embedding model dimensions
    EMBEDDING_DIMENSIONS: dict[str, int] = {
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
    MAX_EMBED_CHARS = 2000

    def __init__(
        self,
        base_url: str | None = None,
        embedding_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        use_cache: bool = True,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.embedding_url = (embedding_url or config.OLLAMA_EMBEDDING_URL).rstrip("/")
        self.api_key = api_key or config.OLLAMA_API_KEY
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.use_cache = use_cache
        self._current_embed_model: str | None = None
        self._current_dimensions: int = 768  # default

        # Connection pooling via session
        self._session: requests.Session | None = None

    @property
    def session(self) -> requests.Session:
        """Lazy-initialized session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
            if self.api_key:
                self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return self._session

    def close(self) -> None:
        """Close the session and release resources."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        self.close()

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
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
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
            response = self.session.post(
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
        """Generate embedding using Ollama with optional caching."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        try:
            if self.use_cache:
                # Use cached version
                embedding = _cached_embedding(text, model, self.embedding_url, self.timeout)
            else:
                # Direct call without cache
                truncated = text[: self.MAX_EMBED_CHARS] if len(text) > self.MAX_EMBED_CHARS else text
                response = self.session.post(
                    f"{self.embedding_url}/api/embed",
                    json={"model": model, "input": truncated},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if not embeddings or not embeddings[0]:
                    raise ValueError("Empty embedding returned from Ollama")
                embedding = tuple(embeddings[0])

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
        """Generate embeddings for multiple texts in a single API call.

        The /api/embed endpoint supports batch inputs natively.
        """
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        # Truncate oversized inputs
        truncated_texts = [text[: self.MAX_EMBED_CHARS] if len(text) > self.MAX_EMBED_CHARS else text for text in texts]

        try:
            response = self.session.post(
                f"{self.embedding_url}/api/embed",
                json={"model": model, "input": truncated_texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            embeddings_list = data.get("embeddings", [])

            if not embeddings_list:
                raise ValueError("Empty embeddings returned from Ollama")

            results = []
            for embedding_data in embeddings_list:
                embedding = tuple(embedding_data) if embedding_data else ()
                if embedding:
                    self._current_dimensions = len(embedding)

                results.append(
                    EmbeddingResponse(
                        embedding=embedding,
                        model=model,
                        provider=self.provider_name,
                        dimensions=len(embedding),
                    )
                )
            return results
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama batch embed failed: {e}") from e

    async def embed_batch_async(
        self,
        texts: list[str],
        model: str,
        max_concurrent: int = 10,  # kept for API compatibility, no longer used
    ) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts asynchronously.

        The /api/embed endpoint supports batch inputs natively, so this
        makes a single async HTTP request for all texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        model : str
            Embedding model name.
        max_concurrent : int
            Deprecated, kept for API compatibility. No longer used since
            the API now supports native batching.

        Returns
        -------
        list[EmbeddingResponse]
            Embeddings in the same order as input texts.

        Examples
        --------
        >>> import trio
        >>> embeddings = trio.run(provider.embed_batch_async, texts, "mxbai-embed-large")
        """
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        # Truncate oversized inputs
        truncated_texts = [text[: self.MAX_EMBED_CHARS] if len(text) > self.MAX_EMBED_CHARS else text for text in texts]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.embedding_url}/api/embed",
                    json={"model": model, "input": truncated_texts},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

            embeddings_list = data.get("embeddings", [])
            if not embeddings_list:
                raise ValueError("Empty embeddings returned from Ollama")

            results = []
            for embedding_data in embeddings_list:
                embedding = tuple(embedding_data) if embedding_data else ()
                if embedding:
                    self._current_dimensions = len(embedding)

                results.append(
                    EmbeddingResponse(
                        embedding=embedding,
                        model=model,
                        provider=self.provider_name,
                        dimensions=len(embedding),
                    )
                )
            return results
        except httpx.HTTPError as e:
            raise ConnectionError(f"Ollama async batch embed failed: {e}") from e

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
            response = self.session.post(
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

    @staticmethod
    def clear_embedding_cache() -> None:
        """Clear the embedding cache."""
        _cached_embedding.cache_clear()

    @staticmethod
    def embedding_cache_info() -> dict[str, int]:
        """Get embedding cache statistics."""
        info = _cached_embedding.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize or 0,
            "currsize": info.currsize,
        }


# Export the EMBEDDING_DIMENSIONS for external use
EMBEDDING_DIMENSIONS = OllamaProvider.EMBEDDING_DIMENSIONS
