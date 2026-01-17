#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
SentenceTransformers provider for offline embedding.

This module provides embedding capabilities using the sentence-transformers
library, enabling fully offline RAG pipelines without API dependencies.

Requires: pip install ragit[transformers]
"""

from typing import TYPE_CHECKING

from ragit.providers.base import (
    BaseEmbeddingProvider,
    EmbeddingResponse,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Lazy import flag
_sentence_transformers_available: bool | None = None
_model_cache: dict[str, "SentenceTransformer"] = {}


def _check_sentence_transformers() -> bool:
    """Check if sentence-transformers is available."""
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
    return _sentence_transformers_available


def _get_model(model_name: str, device: str | None = None) -> "SentenceTransformer":
    """Get or create a cached SentenceTransformer model."""
    cache_key = f"{model_name}:{device or 'auto'}"
    if cache_key not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    return _model_cache[cache_key]


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Embedding provider using sentence-transformers for offline operation.

    This provider uses the sentence-transformers library to generate embeddings
    locally without requiring any API calls. It's ideal for:
    - Offline/air-gapped environments
    - Development and testing
    - Cost-sensitive applications
    - Privacy-sensitive use cases

    Parameters
    ----------
    model_name : str
        HuggingFace model name. Default: "all-MiniLM-L6-v2" (fast, 384 dims).
        Other popular options:
        - "all-mpnet-base-v2" (768 dims, higher quality)
        - "paraphrase-MiniLM-L6-v2" (384 dims)
        - "multi-qa-MiniLM-L6-cos-v1" (384 dims, optimized for QA)
    device : str, optional
        Device to run on ("cpu", "cuda", "mps"). Auto-detected if None.

    Examples
    --------
    >>> # Basic usage
    >>> from ragit.providers import SentenceTransformersProvider
    >>> provider = SentenceTransformersProvider()
    >>>
    >>> # With RAGAssistant (retrieval-only)
    >>> assistant = RAGAssistant(docs, provider=provider)
    >>> results = assistant.retrieve("query")
    >>>
    >>> # Custom model
    >>> provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")

    Raises
    ------
    ImportError
        If sentence-transformers is not installed.

    Note
    ----
    Install with: pip install ragit[transformers]
    """

    # Known model dimensions for common models
    MODEL_DIMENSIONS: dict[str, int] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "all-distilroberta-v1": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        if not _check_sentence_transformers():
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersProvider. "
                "Install with: pip install ragit[transformers]"
            )

        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None  # Lazy loaded
        self._dimensions: int | None = self.MODEL_DIMENSIONS.get(model_name)

    def _ensure_model(self) -> "SentenceTransformer":
        """Ensure model is loaded (lazy loading)."""
        if self._model is None:
            model = _get_model(self._model_name, self._device)
            self._model = model
            # Update dimensions from actual model
            self._dimensions = model.get_sentence_embedding_dimension()
        return self._model

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Load model to get dimensions
            self._ensure_model()
        return self._dimensions or 384  # Fallback

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model_name

    def is_available(self) -> bool:
        """Check if sentence-transformers is installed and model can be loaded."""
        if not _check_sentence_transformers():
            return False
        try:
            self._ensure_model()
            return True
        except Exception:
            return False

    def embed(self, text: str, model: str = "") -> EmbeddingResponse:
        """
        Generate embedding for text.

        Parameters
        ----------
        text : str
            Text to embed.
        model : str
            Model identifier (ignored, uses model from constructor).

        Returns
        -------
        EmbeddingResponse
            The embedding response.
        """
        model_instance = self._ensure_model()
        embedding = model_instance.encode(text, convert_to_numpy=True)

        # Convert to tuple
        embedding_tuple = tuple(float(x) for x in embedding)

        return EmbeddingResponse(
            embedding=embedding_tuple,
            model=self._model_name,
            provider=self.provider_name,
            dimensions=len(embedding_tuple),
        )

    def embed_batch(self, texts: list[str], model: str = "") -> list[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts efficiently.

        Uses batch encoding for better performance.

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
        if not texts:
            return []

        model_instance = self._ensure_model()

        # Batch encode for efficiency
        embeddings = model_instance.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        results = []
        for embedding in embeddings:
            embedding_tuple = tuple(float(x) for x in embedding)
            results.append(
                EmbeddingResponse(
                    embedding=embedding_tuple,
                    model=self._model_name,
                    provider=self.provider_name,
                    dimensions=len(embedding_tuple),
                )
            )

        return results
