#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Function-based provider adapter for pluggable embedding and LLM functions.

This module provides a simple adapter that wraps user-provided functions
into the provider interface, enabling easy integration with custom
embedding and LLM implementations.
"""

import inspect
from collections.abc import Callable

from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)


class FunctionProvider(BaseLLMProvider, BaseEmbeddingProvider):
    """
    Adapter that wraps user-provided embedding and generation functions.

    This provider allows users to bring their own embedding and/or LLM functions
    without implementing the full provider interface.

    Parameters
    ----------
    embed_fn : Callable[[str], list[float]], optional
        Function that takes text and returns an embedding vector.
        Example: `lambda text: openai.embeddings.create(input=text).data[0].embedding`
    generate_fn : Callable, optional
        Function for text generation. Supports two signatures:
        - (prompt: str) -> str
        - (prompt: str, system_prompt: str) -> str
    embedding_dimensions : int, optional
        Embedding dimensions. Auto-detected on first call if not provided.

    Examples
    --------
    >>> # Simple embedding function
    >>> def my_embed(text: str) -> list[float]:
    ...     return openai.embeddings.create(input=text).data[0].embedding
    >>>
    >>> # Use with RAGAssistant (retrieval-only)
    >>> assistant = RAGAssistant(docs, embed_fn=my_embed)
    >>> results = assistant.retrieve("query")
    >>>
    >>> # With LLM for full RAG
    >>> def my_llm(prompt: str, system_prompt: str = None) -> str:
    ...     return openai.chat.completions.create(
    ...         messages=[{"role": "user", "content": prompt}]
    ...     ).choices[0].message.content
    >>>
    >>> assistant = RAGAssistant(docs, embed_fn=my_embed, generate_fn=my_llm)
    >>> answer = assistant.ask("What is X?")
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        generate_fn: Callable[..., str] | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._generate_fn = generate_fn
        self._embedding_dimensions = embedding_dimensions
        self._generate_fn_signature: int | None = None  # Number of args (1 or 2)

        # Detect generate_fn signature if provided
        if generate_fn is not None:
            self._detect_generate_signature()

    def _detect_generate_signature(self) -> None:
        """Detect whether generate_fn accepts 1 or 2 arguments."""
        if self._generate_fn is None:
            return

        sig = inspect.signature(self._generate_fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        # Count required parameters
        required_count = len(params)

        if required_count == 1:
            self._generate_fn_signature = 1
        else:
            # Assume 2 args if more than 1 required or if has optional args
            self._generate_fn_signature = 2

    @property
    def provider_name(self) -> str:
        return "function"

    @property
    def dimensions(self) -> int:
        if self._embedding_dimensions is None:
            raise ValueError("Embedding dimensions not yet determined. Call embed() first or provide dimensions.")
        return self._embedding_dimensions

    @property
    def has_embedding(self) -> bool:
        """Check if embedding function is configured."""
        return self._embed_fn is not None

    @property
    def has_llm(self) -> bool:
        """Check if LLM generation function is configured."""
        return self._generate_fn is not None

    def is_available(self) -> bool:
        """Check if the provider has at least one function configured."""
        return self._embed_fn is not None or self._generate_fn is not None

    def embed(self, text: str, model: str = "") -> EmbeddingResponse:
        """
        Generate embedding using the provided function.

        Parameters
        ----------
        text : str
            Text to embed.
        model : str
            Model identifier (ignored, kept for interface compatibility).

        Returns
        -------
        EmbeddingResponse
            The embedding response.

        Raises
        ------
        ValueError
            If no embedding function was provided.
        """
        if self._embed_fn is None:
            raise ValueError("No embedding function configured. Provide embed_fn to use embeddings.")

        raw_embedding = self._embed_fn(text)

        # Convert to tuple for immutability
        embedding_tuple: tuple[float, ...] = tuple(raw_embedding)

        # Auto-detect dimensions on first call
        if self._embedding_dimensions is None:
            self._embedding_dimensions = len(embedding_tuple)

        return EmbeddingResponse(
            embedding=embedding_tuple,
            model=model or "function",
            provider=self.provider_name,
            dimensions=len(embedding_tuple),
        )

    def embed_batch(self, texts: list[str], model: str = "") -> list[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts.

        Iterates over embed_fn for each text. For providers with native batch
        support, users should implement their own BatchEmbeddingProvider.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        model : str
            Model identifier (ignored).

        Returns
        -------
        list[EmbeddingResponse]
            List of embedding responses.
        """
        return [self.embed(text, model) for text in texts]

    def generate(
        self,
        prompt: str,
        model: str = "",
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Generate text using the provided function.

        Parameters
        ----------
        prompt : str
            The user prompt.
        model : str
            Model identifier (ignored, kept for interface compatibility).
        system_prompt : str, optional
            System prompt for context.
        temperature : float
            Sampling temperature (ignored if function doesn't support it).
        max_tokens : int, optional
            Maximum tokens (ignored if function doesn't support it).

        Returns
        -------
        LLMResponse
            The generated response.

        Raises
        ------
        NotImplementedError
            If no generation function was provided.
        """
        if self._generate_fn is None:
            raise NotImplementedError(
                "No LLM configured. Provide generate_fn or a provider with LLM support "
                "to use ask(), generate(), or generate_code() methods."
            )

        # Call with appropriate signature
        if self._generate_fn_signature == 1:
            # Single argument - prepend system prompt to prompt if provided
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            text = self._generate_fn(full_prompt)
        else:
            # Two arguments - pass separately
            text = self._generate_fn(prompt, system_prompt)

        return LLMResponse(
            text=text,
            model=model or "function",
            provider=self.provider_name,
            usage=None,
        )
