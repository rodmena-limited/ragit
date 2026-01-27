# ragit

RAG toolkit for Python. Document loading, chunking, vector search, LLM integration.

## Installation

```bash
pip install ragit
```

## Quick Start

You must provide an embedding source: custom function, Ollama, or any provider.

### Custom Embedding Function

```python
from ragit import RAGAssistant

def my_embed(text: str) -> list[float]:
    # Use any embedding API: OpenAI, Cohere, HuggingFace, etc.
    return embedding_vector

assistant = RAGAssistant("docs/", embed_fn=my_embed)
results = assistant.retrieve("search query")
```

### With LLM for Q&A

```python
def my_embed(text: str) -> list[float]:
    return embedding_vector

def my_generate(prompt: str, system_prompt: str = "") -> str:
    return llm_response

assistant = RAGAssistant("docs/", embed_fn=my_embed, generate_fn=my_generate)
answer = assistant.ask("How does authentication work?")
```

### With Ollama (nomic-embed-text)

```python
from ragit import RAGAssistant
from ragit.providers import OllamaProvider

# Uses nomic-embed-text for embeddings (768d)
assistant = RAGAssistant("docs/", provider=OllamaProvider())
results = assistant.retrieve("search query")
```

## Core API

```python
assistant = RAGAssistant(
    documents,           # Path, list of Documents, or list of Chunks
    embed_fn=...,        # Embedding function: (str) -> list[float]
    generate_fn=...,     # LLM function: (prompt, system_prompt) -> str
    provider=...,        # Or use a provider instead of functions
    chunk_size=512,
    chunk_overlap=50
)

results = assistant.retrieve(query, top_k=3)      # [(Chunk, score), ...]
context = assistant.get_context(query, top_k=3)   # Formatted string
answer = assistant.ask(question, top_k=3)         # Requires generate_fn/LLM
code = assistant.generate_code(request)           # Requires generate_fn/LLM
```

## Document Loading

```python
from ragit import load_text, load_directory, chunk_text

doc = load_text("file.md")
docs = load_directory("docs/", "*.md")
chunks = chunk_text(text, chunk_size=512, chunk_overlap=50, doc_id="id")
```

## Hyperparameter Optimization

```python
from ragit import RagitExperiment, Document, BenchmarkQuestion

def my_embed(text: str) -> list[float]:
    return embedding_vector

def my_generate(prompt: str, system_prompt: str = "") -> str:
    return llm_response

docs = [Document(id="1", content="...")]
benchmark = [BenchmarkQuestion(question="...", ground_truth="...")]

experiment = RagitExperiment(
    docs, benchmark,
    embed_fn=my_embed,
    generate_fn=my_generate
)
results = experiment.run(max_configs=20)
print(results[0])  # Best config
```

## License

Apache-2.0 - RODMENA LIMITED
