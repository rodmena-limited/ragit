# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2025-01-17

### Breaking Changes

**RAGAssistant and RagitExperiment now require explicit embedding configuration.**

Ragit no longer defaults to Ollama. Users must explicitly configure their embedding and LLM providers.

#### Migration Guide

**Before (v0.7.x):**
```python
from ragit import RAGAssistant, RagitExperiment

# This implicitly used Ollama
assistant = RAGAssistant("docs/")
experiment = RagitExperiment(docs, benchmark)
```

**After (v0.8.0):**
```python
from ragit import RAGAssistant, RagitExperiment

# Option 1: Use your own embedding function (retrieval-only)
def my_embed(text: str) -> list[float]:
    # Your embedding implementation
    return openai.embeddings.create(input=text).data[0].embedding

assistant = RAGAssistant("docs/", embed_fn=my_embed)
results = assistant.retrieve("query")  # Works!
# assistant.ask("question")  # Raises NotImplementedError (no LLM)

# Option 2: Use embedding + generation functions (full RAG)
def my_llm(prompt: str, system_prompt: str = None) -> str:
    # Your LLM implementation
    return openai.chat.completions.create(...).choices[0].message.content

assistant = RAGAssistant("docs/", embed_fn=my_embed, generate_fn=my_llm)
answer = assistant.ask("question")  # Works!

# Option 3: Use sentence-transformers for offline embedding
# pip install ragit[transformers]
from ragit.providers import SentenceTransformersProvider

assistant = RAGAssistant("docs/", provider=SentenceTransformersProvider())

# Option 4: Explicit Ollama (same behavior as before)
from ragit.providers import OllamaProvider

assistant = RAGAssistant("docs/", provider=OllamaProvider())
experiment = RagitExperiment(docs, benchmark, provider=OllamaProvider())
```

### Added

- **FunctionProvider**: New adapter that wraps user-provided embedding and LLM functions
  - Supports single-arg `(prompt)` or two-arg `(prompt, system_prompt)` generation functions
  - Auto-detects embedding dimensions on first call
  - Enables easy integration with any embedding/LLM API

- **SentenceTransformersProvider**: New provider for offline embedding using sentence-transformers
  - Supports popular models like `all-MiniLM-L6-v2` (default), `all-mpnet-base-v2`, etc.
  - Model caching for efficient reuse
  - Batch embedding support
  - Install with: `pip install ragit[transformers]`

- **Retrieval-only mode**: RAGAssistant now supports retrieval without LLM
  - `retrieve()` and `get_context()` work without LLM configuration
  - `ask()`, `generate()`, and `generate_code()` raise `NotImplementedError` if no LLM is configured

- **`has_llm` property**: New property on RAGAssistant to check if LLM is configured

- **Base class exports**: `BaseLLMProvider`, `BaseEmbeddingProvider`, `LLMResponse`, `EmbeddingResponse` are now exported from `ragit.providers`

### Changed

- **RAGAssistant constructor**: Now accepts `embed_fn`, `generate_fn`, and `provider` parameters
  - Must provide either `embed_fn` or `provider`
  - Raises `ValueError` if neither is provided

- **RagitExperiment constructor**: Now accepts `embed_fn`, `generate_fn`, and `provider` parameters
  - Must provide either `embed_fn` or `provider`
  - LLM is required for evaluation (must provide `generate_fn` or a provider with LLM support)

- **Config defaults**: `DEFAULT_LLM_MODEL` and `DEFAULT_EMBEDDING_MODEL` are no longer set by default
  - These can still be configured via environment variables for users who need them

### Removed

- Implicit Ollama fallback when no provider is specified

## [0.7.5] - Previous Release

See git history for changes in previous versions.
