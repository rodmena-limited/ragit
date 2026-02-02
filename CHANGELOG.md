# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2025-02-02

### Added

- **Index persistence**: Save and load indexes to/from disk
  - `save_index(path)` - saves chunks (JSON) + embeddings (numpy binary) + metadata
  - `load_index(path, provider)` - restores a saved index with validation
  - Enables caching expensive embedding computations

- **Thread-safe operations**: RAGAssistant now uses lock-free atomic operations
  - Immutable `IndexState` dataclass holds all index data
  - Reference swaps are atomic under Python's GIL
  - Safe to call `retrieve()` while another thread calls `add_documents()`

- **Context manager for OllamaProvider**: Guaranteed resource cleanup
  ```python
  with OllamaProvider() as provider:
      result = provider.generate("Hello", model="llama3")
  # Session automatically closed
  ```

- **New properties on RAGAssistant**:
  - `is_indexed` - check if index has any documents
  - `chunk_count` - number of chunks in index (alias for `num_chunks`)

- **Embedding count validation**: Raises `IndexingError` if embedding count doesn't match chunk count, preventing silent index corruption

- **Empty document handling**: Logs warning when no chunks are produced from documents

### Changed

- **RAGAssistant is now thread-safe** (previously documented as NOT thread-safe)
- API key no longer stored in session headers (security improvement)
- Docstring examples updated to use `asyncio` instead of `trio`

### Removed

- `scikit-learn` dependency (was unused)
- `pandas` dependency (was unused)
- `trio` dependency (async methods work with `asyncio.run()`)
- `handle_missing_values_in_combinations()` from utils (unused)

### Security

- **API key exposure fix**: API keys are now injected per-request via `_get_headers()` rather than stored in session headers, preventing potential exposure in logs or error messages

## [0.10.0] - 2025-01-27

### Breaking Changes

**Exception types have changed.** OllamaProvider now raises custom exceptions instead of `ConnectionError`.

#### Migration Guide

**Before (v0.9.x):**
```python
from ragit.providers import OllamaProvider

provider = OllamaProvider()
try:
    response = provider.generate("prompt", "model")
except ConnectionError as e:
    print(f"Provider failed: {e}")

try:
    embedding = provider.embed("text", "model")
except ConnectionError as e:
    print(f"Embedding failed: {e}")
```

**After (v0.10.0):**
```python
from ragit.providers import OllamaProvider
from ragit import ProviderError, IndexingError, RagitError

provider = OllamaProvider()

# Option 1: Catch specific exceptions
try:
    response = provider.generate("prompt", "model")
except ProviderError as e:
    print(f"Provider failed: {e}")
    # Access original exception if needed
    if e.original_exception:
        print(f"Caused by: {e.original_exception}")

try:
    embedding = provider.embed("text", "model")
except IndexingError as e:
    print(f"Embedding failed: {e}")

# Option 2: Catch all ragit exceptions
try:
    response = provider.generate("prompt", "model")
except RagitError as e:
    print(f"Ragit error: {e}")
```

**Exception mapping:**
| Old Exception | New Exception | When Raised |
|---------------|---------------|-------------|
| `ConnectionError` | `ProviderError` | LLM generate/chat failures, list_models |
| `ConnectionError` | `IndexingError` | Embedding failures (embed, embed_batch) |

### Added

- **Resilience features** via `resilient-circuit` library:
  - Retry with exponential backoff (3 retries for generate, 2 for embed)
  - Circuit breaker pattern for fault tolerance
  - Per-operation timeouts (generate: 300s, embed: 30s, embed_batch: 120s)
  - `use_resilience` parameter to enable/disable (default: True)
  - `generate_circuit_status` and `embed_circuit_status` properties

- **Custom exception hierarchy** (`ragit.exceptions`):
  - `RagitError` - base exception for all ragit errors
  - `ConfigurationError` - configuration validation failures
  - `ProviderError` - provider communication failures
  - `IndexingError` - embedding/indexing failures
  - `RetrievalError` - retrieval operation failures
  - `GenerationError` - LLM generation failures
  - `EvaluationError` - evaluation/scoring failures
  - `ExceptionAggregator` - collect errors during batch operations

- **Structured logging** (`ragit.logging`):
  - `setup_logging()` - configure ragit logging
  - `log_operation()` - context manager with timing
  - `log_method()` - decorator for method logging
  - `LogContext` - request correlation tracking

- **Execution monitoring** (`ragit.monitor`):
  - `ExecutionMonitor` - track pattern/step execution times
  - JSON export for analysis
  - Step aggregates (count, total, avg, min, max)

- **Window search / context expansion** on `RAGAssistant`:
  - `retrieve_with_context(query, top_k, window_size)` - expand retrieval with adjacent chunks
  - `get_context_with_window(query, top_k, window_size)` - merged context string
  - `min_score` parameter for score threshold filtering

- **Rich chunk metadata**:
  - `generate_document_id(content)` - SHA256-based document ID
  - `deduplicate_documents(docs)` - remove duplicates by content hash
  - Chunks now include: `document_id`, `sequence_number`, `chunk_start`, `chunk_end`

- **Pydantic configuration validation** (`ragit.config`):
  - `RagitConfig` model with field validators
  - URL format validation
  - Timeout bounds checking (1-600 seconds)
  - `ConfigValidationError` for invalid configuration

### Changed

- `OllamaProvider` now uses custom exceptions instead of `ConnectionError`
- Configuration uses Pydantic validation (stricter validation at startup)
- All chunking functions now include rich metadata by default
- New dependency: `resilient-circuit>=0.4.7`

### Fixed

- **Cache truncation bug**: Text is now truncated BEFORE cache key lookup, ensuring consistent caching behavior for long texts

## [0.9.0] - 2025-01-27

### Breaking Changes

**SentenceTransformersProvider has been removed.**

The offline MiniLM/sentence-transformers embedding provider has been removed. Use OllamaProvider with `nomic-embed-text` for embeddings instead.

#### Migration Guide

**Before (v0.8.x):**
```python
from ragit.providers import SentenceTransformersProvider
assistant = RAGAssistant("docs/", provider=SentenceTransformersProvider())
```

**After (v0.9.0):**
```python
from ragit.providers import OllamaProvider
assistant = RAGAssistant("docs/", provider=OllamaProvider())
# Uses nomic-embed-text (768d) for embeddings
```

### Removed

- `SentenceTransformersProvider` - Use `OllamaProvider` with nomic-embed-text instead
- `ragit[transformers]` optional dependency - No longer needed
- `sentence-transformers` dependency

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
