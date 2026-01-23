#!/usr/bin/env python3
#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Example: Writing a Custom Provider

This example shows how to create custom providers for ragit.
We demonstrate an Ollama provider as a reference implementation.

To use ragit with any embedding/LLM API, you can either:
1. Use simple functions (embed_fn, generate_fn) - easiest approach
2. Create a provider class - more control, caching, connection pooling

Usage:
    python demos/custom_provider_example.py
"""

import requests

from ragit import RAGAssistant
from ragit.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    EmbeddingResponse,
    LLMResponse,
)

# =============================================================================
# Option 1: Simple Functions (Recommended for most use cases)
# =============================================================================


def openai_embed(text: str) -> list[float]:
    """Example embedding function using OpenAI API."""
    import openai

    response = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def openai_generate(prompt: str, system_prompt: str = "") -> str:
    """Example generation function using OpenAI API."""
    import openai

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content


# Usage with functions:
# assistant = RAGAssistant("docs/", embed_fn=openai_embed, generate_fn=openai_generate)


# =============================================================================
# Option 2: Custom Provider Class (For more control)
# =============================================================================


class OllamaProvider(BaseEmbeddingProvider, BaseLLMProvider):
    """
    Example provider implementation for Ollama.

    This shows how to implement both BaseEmbeddingProvider and BaseLLMProvider.
    You can implement just one if you only need embedding or LLM.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.timeout = timeout
        self._session: requests.Session | None = None
        self._dimensions: int = 768  # Default, updated on first embed

    @property
    def session(self) -> requests.Session:
        """Lazy session initialization for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Content-Type": "application/json"})
        return self._session

    # -------------------------------------------------------------------------
    # BaseEmbeddingProvider implementation
    # -------------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Unique identifier for this provider."""
        return "ollama"

    @property
    def dimensions(self) -> int:
        """Embedding dimensions (updated after first embed call)."""
        return self._dimensions

    def embed(self, text: str, model: str | None = None) -> EmbeddingResponse:
        """Generate embedding for a single text."""
        # Use provider's model if none specified or if "default"
        model = model if model and model != "default" else self.embedding_model

        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        embedding = tuple(data["embedding"])
        self._dimensions = len(embedding)

        return EmbeddingResponse(
            embedding=embedding,
            model=model,
            provider=self.provider_name,
            dimensions=self._dimensions,
        )

    def embed_batch(self, texts: list[str], model: str | None = None) -> list[EmbeddingResponse]:
        """Batch embedding - iterate over texts."""
        return [self.embed(text, model) for text in texts]

    def is_available(self) -> bool:
        """Check if provider is reachable."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # -------------------------------------------------------------------------
    # BaseLLMProvider implementation
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text using the LLM."""
        model = model or self.llm_model

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()

        return LLMResponse(
            text=data["response"],
            model=model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )


# =============================================================================
# Demo
# =============================================================================


def main() -> None:
    """Demonstrate custom provider usage."""
    print("Custom Provider Example")
    print("=" * 50)

    # Check if Ollama is available
    provider = OllamaProvider()

    if not provider.is_available():
        print("Ollama not available. Showing usage patterns instead.\n")
        print("With custom provider:")
        print('  assistant = RAGAssistant("docs/", provider=OllamaProvider())')
        print()
        print("With custom functions:")
        print('  assistant = RAGAssistant("docs/", embed_fn=my_embed, generate_fn=my_llm)')
        return

    print(f"Ollama available at {provider.base_url}")

    # Test embedding
    print("\nTesting embedding...")
    response = provider.embed("Hello, world!")
    print(f"  Dimensions: {response.dimensions}")
    print(f"  First 5 values: {response.embedding[:5]}")

    # Test generation (optional - requires LLM model)
    print("\nTesting generation...")
    try:
        response = provider.generate(
            "What is 2+2? Answer in one word.",
            temperature=0.1,
        )
        print(f"  Response: {response.text.strip()}")
    except Exception as e:
        print(f"  Skipped (no LLM model available): {e}")

    # Use with RAGAssistant
    print("\nUsing with RAGAssistant...")
    assistant = RAGAssistant(
        "demos/falcon-tutorial.rst",
        provider=provider,
    )
    print(f"  Loaded {assistant.num_chunks} chunks")

    results = assistant.retrieve("How do I create a REST API?", top_k=2)
    print(f"  Found {len(results)} relevant chunks")
    for chunk, score in results:
        print(f"    Score {score:.3f}: {chunk.content[:60]}...")


if __name__ == "__main__":
    main()
