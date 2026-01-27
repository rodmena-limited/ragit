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

Resilience features (via resilient-circuit):
- Retry with exponential backoff
- Circuit breaker pattern for fault tolerance
"""

from datetime import timedelta
from fractions import Fraction
from functools import lru_cache
from typing import Any

import httpx
import requests
from resilient_circuit import (
    CircuitProtectorPolicy,
    ExponentialDelay,
    RetryWithBackoffPolicy,
    SafetyNet,
)
from resilient_circuit.exceptions import ProtectedCallError, RetryLimitReached

from ragit.config import config
from ragit.exceptions import IndexingError, ProviderError
from ragit.logging import log_operation, logger
from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)


def _create_generate_policy() -> SafetyNet:
    """Create resilience policy for LLM generation (longer timeouts, more tolerant)."""
    return SafetyNet(
        policies=(
            RetryWithBackoffPolicy(
                max_retries=3,
                backoff=ExponentialDelay(
                    min_delay=timedelta(seconds=1),
                    max_delay=timedelta(seconds=30),
                    factor=2,
                    jitter=0.1,
                ),
                should_handle=lambda e: isinstance(e, (ConnectionError, TimeoutError, requests.RequestException)),
            ),
            CircuitProtectorPolicy(
                resource_key="ollama_generate",
                cooldown=timedelta(seconds=60),
                failure_limit=Fraction(3, 10),  # 30% failure rate trips circuit
                success_limit=Fraction(4, 5),  # 80% success to close
                should_handle=lambda e: isinstance(e, (ConnectionError, requests.RequestException)),
            ),
        )
    )


def _create_embed_policy() -> SafetyNet:
    """Create resilience policy for embeddings (faster, stricter)."""
    return SafetyNet(
        policies=(
            RetryWithBackoffPolicy(
                max_retries=2,
                backoff=ExponentialDelay(
                    min_delay=timedelta(milliseconds=500),
                    max_delay=timedelta(seconds=5),
                    factor=2,
                    jitter=0.1,
                ),
                should_handle=lambda e: isinstance(e, (ConnectionError, TimeoutError, requests.RequestException)),
            ),
            CircuitProtectorPolicy(
                resource_key="ollama_embed",
                cooldown=timedelta(seconds=30),
                failure_limit=Fraction(2, 5),  # 40% failure rate trips circuit
                success_limit=Fraction(3, 3),  # All 3 tests must succeed to close
                should_handle=lambda e: isinstance(e, (ConnectionError, requests.RequestException)),
            ),
        )
    )


def _truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars. Used BEFORE cache lookup to fix cache key bug."""
    return text[:max_chars] if len(text) > max_chars else text


# Module-level cache for embeddings (shared across instances)
# NOTE: Text must be truncated BEFORE calling this function to ensure correct cache keys
@lru_cache(maxsize=2048)
def _cached_embedding(text: str, model: str, embedding_url: str, timeout: int) -> tuple[float, ...]:
    """Cache embedding results to avoid redundant API calls.

    IMPORTANT: Caller must truncate text BEFORE calling this function.
    This ensures cache keys are consistent for truncated inputs.
    """
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

    # Default timeouts per operation type (in seconds)
    DEFAULT_TIMEOUTS: dict[str, int] = {
        "generate": 300,  # 5 minutes for LLM generation
        "chat": 300,  # 5 minutes for chat
        "embed": 30,  # 30 seconds for single embedding
        "embed_batch": 120,  # 2 minutes for batch embedding
        "health": 5,  # 5 seconds for health check
        "list_models": 10,  # 10 seconds for listing models
    }

    def __init__(
        self,
        base_url: str | None = None,
        embedding_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        timeouts: dict[str, int] | None = None,
        use_cache: bool = True,
        use_resilience: bool = True,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.embedding_url = (embedding_url or config.OLLAMA_EMBEDDING_URL).rstrip("/")
        self.api_key = api_key or config.OLLAMA_API_KEY
        self.use_cache = use_cache
        self.use_resilience = use_resilience
        self._current_embed_model: str | None = None
        self._current_dimensions: int = 768  # default

        # Per-operation timeouts (merge user overrides with defaults)
        self._timeouts = {**self.DEFAULT_TIMEOUTS, **(timeouts or {})}
        # Legacy single timeout parameter overrides all operations
        if timeout is not None:
            self._timeouts = {k: timeout for k in self._timeouts}
        # Keep legacy timeout property for backwards compatibility
        self.timeout = timeout or config.OLLAMA_TIMEOUT

        # Connection pooling via session
        self._session: requests.Session | None = None

        # Resilience policies (retry + circuit breaker)
        self._generate_policy: SafetyNet | None = None
        self._embed_policy: SafetyNet | None = None
        if use_resilience:
            self._generate_policy = _create_generate_policy()
            self._embed_policy = _create_embed_policy()

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
                timeout=self._timeouts["health"],
            )
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=self._timeouts["list_models"],
            )
            response.raise_for_status()
            data = response.json()
            return list(data.get("models", []))
        except requests.RequestException as e:
            raise ProviderError("Failed to list Ollama models", e) from e

    def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text using Ollama with optional resilience (retry + circuit breaker)."""
        if self.use_resilience and self._generate_policy is not None:

            @self._generate_policy
            def _protected_generate() -> LLMResponse:
                return self._do_generate(prompt, model, system_prompt, temperature, max_tokens)

            try:
                return _protected_generate()
            except ProtectedCallError as e:
                logger.warning(f"Circuit breaker OPEN for ollama.generate (model={model})")
                raise ProviderError("Ollama service unavailable - circuit breaker open", e) from e
            except RetryLimitReached as e:
                logger.error(f"Retry limit reached for ollama.generate (model={model}): {e.__cause__}")
                raise ProviderError("Ollama generate failed after retries", e.__cause__) from e
        else:
            return self._do_generate(prompt, model, system_prompt, temperature, max_tokens)

    def _do_generate(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Internal generate implementation (unprotected)."""
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

        with log_operation("ollama.generate", model=model, prompt_len=len(prompt)) as ctx:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self._timeouts["generate"],
                )
                response.raise_for_status()
                data = response.json()

                ctx["completion_tokens"] = data.get("eval_count")

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
                raise ProviderError("Ollama generate failed", e) from e

    def embed(self, text: str, model: str) -> EmbeddingResponse:
        """Generate embedding using Ollama with optional caching and resilience."""
        if self.use_resilience and self._embed_policy is not None:

            @self._embed_policy
            def _protected_embed() -> EmbeddingResponse:
                return self._do_embed(text, model)

            try:
                return _protected_embed()
            except ProtectedCallError as e:
                logger.warning(f"Circuit breaker OPEN for ollama.embed (model={model})")
                raise ProviderError("Ollama embedding service unavailable - circuit breaker open", e) from e
            except RetryLimitReached as e:
                logger.error(f"Retry limit reached for ollama.embed (model={model}): {e.__cause__}")
                raise IndexingError("Ollama embed failed after retries", e.__cause__) from e
        else:
            return self._do_embed(text, model)

    def _do_embed(self, text: str, model: str) -> EmbeddingResponse:
        """Internal embed implementation (unprotected)."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        # Truncate BEFORE cache lookup (fixes cache key bug)
        truncated_text = _truncate_text(text, self.MAX_EMBED_CHARS)
        was_truncated = len(text) > self.MAX_EMBED_CHARS

        with log_operation("ollama.embed", model=model, text_len=len(text), truncated=was_truncated) as ctx:
            try:
                if self.use_cache:
                    # Use cached version with truncated text
                    embedding = _cached_embedding(truncated_text, model, self.embedding_url, self._timeouts["embed"])
                    ctx["cache"] = "hit_or_miss"  # Can't tell from here
                else:
                    # Direct call without cache
                    response = self.session.post(
                        f"{self.embedding_url}/api/embed",
                        json={"model": model, "input": truncated_text},
                        timeout=self._timeouts["embed"],
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    if not embeddings or not embeddings[0]:
                        raise ValueError("Empty embedding returned from Ollama")
                    embedding = tuple(embeddings[0])
                    ctx["cache"] = "disabled"

                # Update dimensions from actual response
                self._current_dimensions = len(embedding)
                ctx["dimensions"] = len(embedding)

                return EmbeddingResponse(
                    embedding=embedding,
                    model=model,
                    provider=self.provider_name,
                    dimensions=len(embedding),
                )
            except requests.RequestException as e:
                raise IndexingError("Ollama embed failed", e) from e

    def embed_batch(self, texts: list[str], model: str) -> list[EmbeddingResponse]:
        """Generate embeddings for multiple texts in a single API call with resilience.

        The /api/embed endpoint supports batch inputs natively.
        """
        if self.use_resilience and self._embed_policy is not None:

            @self._embed_policy
            def _protected_embed_batch() -> list[EmbeddingResponse]:
                return self._do_embed_batch(texts, model)

            try:
                return _protected_embed_batch()
            except ProtectedCallError as e:
                logger.warning(f"Circuit breaker OPEN for ollama.embed_batch (model={model}, batch_size={len(texts)})")
                raise ProviderError("Ollama embedding service unavailable - circuit breaker open", e) from e
            except RetryLimitReached as e:
                logger.error(f"Retry limit reached for ollama.embed_batch (model={model}): {e.__cause__}")
                raise IndexingError("Ollama batch embed failed after retries", e.__cause__) from e
        else:
            return self._do_embed_batch(texts, model)

    def _do_embed_batch(self, texts: list[str], model: str) -> list[EmbeddingResponse]:
        """Internal batch embed implementation (unprotected)."""
        self._current_embed_model = model
        self._current_dimensions = self.EMBEDDING_DIMENSIONS.get(model, 768)

        # Truncate oversized inputs
        truncated_texts = [_truncate_text(text, self.MAX_EMBED_CHARS) for text in texts]
        truncated_count = sum(1 for t, tt in zip(texts, truncated_texts, strict=True) if len(t) != len(tt))

        with log_operation(
            "ollama.embed_batch", model=model, batch_size=len(texts), truncated_count=truncated_count
        ) as ctx:
            try:
                response = self.session.post(
                    f"{self.embedding_url}/api/embed",
                    json={"model": model, "input": truncated_texts},
                    timeout=self._timeouts["embed_batch"],
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

                ctx["dimensions"] = self._current_dimensions
                return results
            except requests.RequestException as e:
                raise IndexingError("Ollama batch embed failed", e) from e

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
                    timeout=self._timeouts["embed_batch"],
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
            raise IndexingError("Ollama async batch embed failed", e) from e

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Chat completion using Ollama with optional resilience.

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
        if self.use_resilience and self._generate_policy is not None:

            @self._generate_policy
            def _protected_chat() -> LLMResponse:
                return self._do_chat(messages, model, temperature, max_tokens)

            try:
                return _protected_chat()
            except ProtectedCallError as e:
                logger.warning(f"Circuit breaker OPEN for ollama.chat (model={model})")
                raise ProviderError("Ollama service unavailable - circuit breaker open", e) from e
            except RetryLimitReached as e:
                logger.error(f"Retry limit reached for ollama.chat (model={model}): {e.__cause__}")
                raise ProviderError("Ollama chat failed after retries", e.__cause__) from e
        else:
            return self._do_chat(messages, model, temperature, max_tokens)

    def _do_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Internal chat implementation (unprotected)."""
        options: dict[str, float | int] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: dict[str, str | bool | list[dict[str, str]] | dict[str, float | int]] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        with log_operation("ollama.chat", model=model, message_count=len(messages)) as ctx:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self._timeouts["chat"],
                )
                response.raise_for_status()
                data = response.json()

                ctx["completion_tokens"] = data.get("eval_count")

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
                raise ProviderError("Ollama chat failed", e) from e

    # Circuit breaker status monitoring
    @property
    def generate_circuit_status(self) -> str:
        """Get generate circuit breaker status (CLOSED, OPEN, HALF_OPEN, or 'disabled')."""
        if not self.use_resilience or self._generate_policy is None:
            return "disabled"
        # Access the circuit protector (second policy in SafetyNet)
        circuit = self._generate_policy._policies[1]
        return circuit.status.name

    @property
    def embed_circuit_status(self) -> str:
        """Get embed circuit breaker status (CLOSED, OPEN, HALF_OPEN, or 'disabled')."""
        if not self.use_resilience or self._embed_policy is None:
            return "disabled"
        circuit = self._embed_policy._policies[1]
        return circuit.status.name

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
