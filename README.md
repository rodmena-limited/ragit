# ragit

A Python toolkit for building Retrieval-Augmented Generation (RAG) applications. Ragit provides document loading, chunking, vector search, and LLM integration out of the box, allowing you to build document Q&A systems and code generators with minimal boilerplate.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Tutorial: Using Ragit](#tutorial-using-ragit)
   - [Loading Documents](#loading-documents)
   - [The RAGAssistant Class](#the-ragassistant-class)
   - [Asking Questions](#asking-questions)
   - [Generating Code](#generating-code)
   - [Custom Retrieval](#custom-retrieval)
4. [Tutorial: Platform Integration](#tutorial-platform-integration)
   - [Flask Integration](#flask-integration)
   - [FastAPI Integration](#fastapi-integration)
   - [Command-Line Tools](#command-line-tools)
   - [Batch Processing](#batch-processing)
5. [Advanced: Hyperparameter Optimization](#advanced-hyperparameter-optimization)
6. [API Reference](#api-reference)
7. [License](#license)

## Installation

```bash
pip install ragit
```

Ragit requires an Ollama-compatible API for embeddings and LLM inference. You can use:
- A local Ollama instance (https://ollama.ai)
- A cloud-hosted Ollama API
- Any OpenAI-compatible API endpoint

## Configuration

Ragit reads configuration from environment variables. Create a `.env` file in your project root:

```bash
# LLM API (cloud or local)
OLLAMA_BASE_URL=https://your-ollama-api.com
OLLAMA_API_KEY=your-api-key

# Embedding API (can be different from LLM)
OLLAMA_EMBEDDING_URL=http://localhost:11434

# Default models
RAGIT_DEFAULT_LLM_MODEL=llama3.1:8b
RAGIT_DEFAULT_EMBEDDING_MODEL=mxbai-embed-large
```

A common setup is to use a cloud API for LLM inference (faster, more capable models) while running embeddings locally (lower latency, no API costs for indexing).

## Tutorial: Using Ragit

This section covers the core functionality of ragit: loading documents, creating a RAG assistant, and querying your knowledge base.

### Loading Documents

Ragit provides several functions for loading and chunking documents.

**Loading a single file:**

```python
from ragit import load_text

doc = load_text("docs/api-reference.md")
print(doc.id)       # "api-reference"
print(doc.content)  # Full file contents
```

**Loading a directory:**

```python
from ragit import load_directory

# Load all markdown files
docs = load_directory("docs/", "*.md")

# Load recursively
docs = load_directory("docs/", "**/*.md", recursive=True)

# Load multiple file types
txt_docs = load_directory("docs/", "*.txt")
rst_docs = load_directory("docs/", "*.rst")
all_docs = txt_docs + rst_docs
```

**Custom chunking:**

For fine-grained control over how documents are split:

```python
from ragit import chunk_text, chunk_by_separator, chunk_rst_sections

# Fixed-size chunks with overlap
chunks = chunk_text(
    text,
    chunk_size=512,      # Characters per chunk
    chunk_overlap=50,    # Overlap between chunks
    doc_id="my-doc"
)

# Split by paragraph
chunks = chunk_by_separator(text, separator="\n\n")

# Split RST documents by section headers
chunks = chunk_rst_sections(rst_content, doc_id="tutorial")
```

### The RAGAssistant Class

The `RAGAssistant` class is the main interface for RAG operations. It handles document indexing, retrieval, and generation in a single object.

```python
from ragit import RAGAssistant

# Create from a directory
assistant = RAGAssistant("docs/")

# Create from a single file
assistant = RAGAssistant("docs/tutorial.rst")

# Create from Document objects
from ragit import Document

docs = [
    Document(id="intro", content="Introduction to the API..."),
    Document(id="auth", content="Authentication uses JWT tokens..."),
    Document(id="endpoints", content="Available endpoints: /users, /items..."),
]
assistant = RAGAssistant(docs)
```

**Configuration options:**

```python
assistant = RAGAssistant(
    "docs/",
    embedding_model="mxbai-embed-large",  # Model for embeddings
    llm_model="llama3.1:70b",             # Model for generation
    chunk_size=512,                        # Characters per chunk
    chunk_overlap=50,                      # Overlap between chunks
)
```

### Asking Questions

The `ask()` method retrieves relevant context and generates an answer:

```python
assistant = RAGAssistant("docs/")

answer = assistant.ask("How do I authenticate API requests?")
print(answer)
```

**Customizing the query:**

```python
answer = assistant.ask(
    "How do I authenticate API requests?",
    top_k=5,                    # Number of chunks to retrieve
    temperature=0.3,            # Lower = more focused answers
    system_prompt="You are a technical documentation assistant. "
                  "Answer concisely and include code examples."
)
```

### Generating Code

The `generate_code()` method is optimized for producing clean, runnable code:

```python
assistant = RAGAssistant("framework-docs/")

code = assistant.generate_code(
    "Create a REST API endpoint for user registration",
    language="python"
)
print(code)
```

The output is clean code without markdown formatting. The assistant uses your documentation as context to generate framework-specific, idiomatic code.

### Custom Retrieval

For advanced use cases, you can access the retrieval and generation steps separately:

```python
assistant = RAGAssistant("docs/")

# Step 1: Retrieve relevant chunks
results = assistant.retrieve("authentication", top_k=5)
for chunk, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {chunk.content[:200]}...")
    print()

# Step 2: Get formatted context string
context = assistant.get_context("authentication", top_k=3)

# Step 3: Generate with custom prompt
prompt = f"""Based on this documentation:

{context}

Write a Python function that validates a JWT token."""

response = assistant.generate(
    prompt,
    system_prompt="You are an expert Python developer.",
    temperature=0.2
)
```

## Tutorial: Platform Integration

This section shows how to integrate ragit into web applications and other platforms.

### Flask Integration

```python
from flask import Flask, request, jsonify
from ragit import RAGAssistant

app = Flask(__name__)

# Initialize once at startup
assistant = RAGAssistant("docs/")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "question is required"}), 400

    answer = assistant.ask(question, top_k=3)
    return jsonify({"answer": answer})

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", 5))

    results = assistant.retrieve(query, top_k=top_k)
    return jsonify({
        "results": [
            {"content": chunk.content, "score": score}
            for chunk, score in results
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ragit import RAGAssistant

app = FastAPI()

# Initialize once at startup
assistant = RAGAssistant("docs/")

class Question(BaseModel):
    question: str
    top_k: int = 3
    temperature: float = 0.7

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    answer = assistant.ask(
        q.question,
        top_k=q.top_k,
        temperature=q.temperature
    )
    return Answer(answer=answer)

@app.get("/search")
async def search(q: str, top_k: int = 5):
    results = assistant.retrieve(q, top_k=top_k)
    return {
        "results": [
            {"content": chunk.content, "score": score}
            for chunk, score in results
        ]
    }
```

### Command-Line Tools

Build CLI tools using argparse or click:

```python
#!/usr/bin/env python3
import argparse
from ragit import RAGAssistant

def main():
    parser = argparse.ArgumentParser(description="Query documentation")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--docs", default="docs/", help="Documentation path")
    parser.add_argument("--top-k", type=int, default=3, help="Context chunks")
    args = parser.parse_args()

    assistant = RAGAssistant(args.docs)
    answer = assistant.ask(args.question, top_k=args.top_k)
    print(answer)

if __name__ == "__main__":
    main()
```

Usage:

```bash
python ask.py "How do I configure logging?"
python ask.py "What are the API rate limits?" --docs api-docs/ --top-k 5
```

### Batch Processing

Process multiple questions or generate reports:

```python
from ragit import RAGAssistant

assistant = RAGAssistant("docs/")

questions = [
    "What authentication methods are supported?",
    "How do I handle errors?",
    "What are the rate limits?",
]

# Process questions
results = {}
for question in questions:
    results[question] = assistant.ask(question)

# Generate a report
with open("qa-report.md", "w") as f:
    f.write("# Documentation Q&A Report\n\n")
    for question, answer in results.items():
        f.write(f"## {question}\n\n")
        f.write(f"{answer}\n\n")
```

## Advanced: Hyperparameter Optimization

Ragit includes tools to find the optimal RAG configuration for your specific documents and use case.

```python
from ragit import RagitExperiment, Document, BenchmarkQuestion

# Your documents
documents = [
    Document(id="auth", content="Authentication uses Bearer tokens..."),
    Document(id="api", content="The API supports GET, POST, PUT, DELETE..."),
]

# Benchmark questions with expected answers
benchmark = [
    BenchmarkQuestion(
        question="What authentication method does the API use?",
        ground_truth="The API uses Bearer token authentication."
    ),
    BenchmarkQuestion(
        question="What HTTP methods are supported?",
        ground_truth="GET, POST, PUT, and DELETE methods are supported."
    ),
]

# Run optimization
experiment = RagitExperiment(documents, benchmark)
results = experiment.run(max_configs=20)

# Get the best configuration
best = results[0]
print(f"Best config: chunk_size={best.config.chunk_size}, "
      f"chunk_overlap={best.config.chunk_overlap}, "
      f"top_k={best.config.top_k}")
print(f"Score: {best.score:.3f}")
```

The experiment tests different combinations of chunk sizes, overlaps, and retrieval parameters to find what works best for your content.

## Performance Features

Ragit includes several optimizations for production workloads:

### Connection Pooling

`OllamaProvider` uses HTTP connection pooling via `requests.Session()` for faster sequential requests:

```python
from ragit.providers import OllamaProvider

provider = OllamaProvider()

# All requests reuse the same connection pool
for text in texts:
    provider.embed(text, model="mxbai-embed-large")

# Explicitly close when done (optional, auto-closes on garbage collection)
provider.close()
```

### Async Parallel Embedding

For large batches, use `embed_batch_async()` with trio for 5-10x faster embedding:

```python
import trio
from ragit.providers import OllamaProvider

provider = OllamaProvider()

async def embed_documents():
    texts = ["doc1...", "doc2...", "doc3...", ...]  # hundreds of texts
    embeddings = await provider.embed_batch_async(
        texts,
        model="mxbai-embed-large",
        max_concurrent=10  # Adjust based on server capacity
    )
    return embeddings

# Run with trio
results = trio.run(embed_documents)
```

### Embedding Cache

Repeated embedding calls are cached automatically (2048 entries LRU):

```python
from ragit.providers import OllamaProvider

provider = OllamaProvider(use_cache=True)  # Default

# First call hits the API
provider.embed("Hello world", model="mxbai-embed-large")

# Second call returns cached result instantly
provider.embed("Hello world", model="mxbai-embed-large")

# View cache statistics
print(OllamaProvider.embedding_cache_info())
# {'hits': 1, 'misses': 1, 'maxsize': 2048, 'currsize': 1}

# Clear cache if needed
OllamaProvider.clear_embedding_cache()
```

### Pre-normalized Embeddings

Vector similarity uses pre-normalized embeddings, making cosine similarity a simple dot product (O(1) per comparison).

## API Reference

### Document Loading

| Function | Description |
|----------|-------------|
| `load_text(path)` | Load a single text file as a Document |
| `load_directory(path, pattern, recursive=False)` | Load files matching a glob pattern |
| `chunk_text(text, chunk_size, chunk_overlap, doc_id)` | Split text into overlapping chunks |
| `chunk_document(doc, chunk_size, chunk_overlap)` | Split a Document into chunks |
| `chunk_by_separator(text, separator, doc_id)` | Split text by a delimiter |
| `chunk_rst_sections(text, doc_id)` | Split RST by section headers |

### RAGAssistant

| Method | Description |
|--------|-------------|
| `retrieve(query, top_k=3)` | Return list of (Chunk, score) tuples |
| `get_context(query, top_k=3)` | Return formatted context string |
| `generate(prompt, system_prompt, temperature)` | Generate text without retrieval |
| `ask(question, system_prompt, top_k, temperature)` | Retrieve context and generate answer |
| `generate_code(request, language, top_k, temperature)` | Generate clean code |

### Properties

| Property | Description |
|----------|-------------|
| `assistant.num_documents` | Number of loaded documents |
| `assistant.num_chunks` | Number of indexed chunks |
| `assistant.embedding_model` | Current embedding model |
| `assistant.llm_model` | Current LLM model |

## License

Apache-2.0 - RODMENA LIMITED
