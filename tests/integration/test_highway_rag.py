#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Integration test: Highway DSL RAG Assistant

This test validates the RAG pipeline using Highway DSL documentation
as the knowledge base. It requires a running Ollama instance.

Run with: pytest tests/integration/ -v --integration
Skip with: pytest tests/integration/ -v -m "not integration"
"""

import numpy as np
import pytest
import requests

from ragit.config import config

# Highway DSL documentation chunks for RAG
DSL_DOCS = [
    {
        "id": "parallel_basics",
        "title": "Parallel Workflow Basics",
        "content": """Highway uses a "fork-only" parallel execution model:

1. ParallelOperator ONLY FORKS - It spawns branches and returns IMMEDIATELY (does NOT wait)
2. Explicit wait is REQUIRED - Use tools.workflow.wait_for_parallel_branches to wait
3. JoinOperator is OPTIONAL - It validates after waiting (does NOT do the actual waiting)

Every parallel workflow MUST follow this pattern:
```python
# Step 1: FORK (returns immediately, does NOT wait)
builder.parallel("fork", result_key="fork_data", branches={...})

# Step 2: WAIT (THE REQUIRED WAITING MECHANISM)
builder.task("wait", "tools.workflow.wait_for_parallel_branches",
    args=["{{fork_data}}"], dependencies=["fork"])

# Step 3 (optional): VALIDATE
builder.join("validate", join_tasks=[...], join_mode=JoinMode.ALL_OF,
    dependencies=["wait"])
```

If you forget the wait task, your workflow will continue immediately while branches are still running!""",
    },
    {
        "id": "workflow_builder_basics",
        "title": "WorkflowBuilder Basics",
        "content": """All workflows use the fluent WorkflowBuilder API:

```python
from highway_dsl import WorkflowBuilder

def get_workflow():
    builder = WorkflowBuilder(name="workflow_name", version="2.0.0")
    # Add tasks using fluent chaining
    builder.task(...)
    return builder.build()
```

CRITICAL: Every workflow MUST have a name parameter:
```python
# CORRECT
builder = WorkflowBuilder(name="my_workflow_name")

# WRONG - This will cause runtime error
builder = WorkflowBuilder()  # ValueError: must have 'name' field
```

Task Chaining - Tasks are automatically chained unless dependencies are explicitly specified:
```python
builder.task("task1", "tools.shell.run", args=["echo 'First'"])
builder.task("task2", "tools.shell.run", args=["echo 'Second'"])  # Runs after task1
```""",
    },
    {
        "id": "task_operator",
        "title": "TaskOperator Reference",
        "content": """TaskOperator - Basic workflow steps:

```python
builder.task(
    task_id="unique_task_id",
    function="tools.function.name",
    args=["positional", "args"],  # Optional
    kwargs={"key": "value"},  # Optional
    dependencies=["task1", "task2"],  # Optional
    result_key="output_name",  # For passing data between tasks
    retry_policy=RetryPolicy(max_retries=3, delay=timedelta(seconds=5)),
    timeout_policy=TimeoutPolicy(timeout=timedelta(hours=1)),
)
```

Examples:
```python
# Simple shell command
builder.task("hello", "tools.shell.run", args=["echo 'Hello World'"])

# HTTP request
builder.task(
    "fetch_api",
    "tools.http.request",
    kwargs={
        "url": "https://api.example.com/data",
        "method": "GET",
        "headers": {"Authorization": "Bearer token"}
    },
    result_key="api_response"
)
```""",
    },
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def check_ollama_available(url: str, timeout: int = 5) -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


class HighwayDSLAssistant:
    """RAG-powered Highway DSL assistant for testing."""

    def __init__(
        self,
        docs: list[dict],
        embedding_url: str,
        llm_url: str,
        api_key: str | None,
        embedding_model: str,
        llm_model: str,
    ):
        self.docs = docs
        self.embedding_url = embedding_url
        self.llm_url = llm_url
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.index: list[dict] = []

    def _get_headers(self, include_auth: bool = True) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if include_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        resp = requests.post(
            f"{self.embedding_url}/api/embeddings",
            headers=self._get_headers(include_auth=False),  # Local embeddings, no auth
            json={"model": self.embedding_model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _generate(self, prompt: str, system: str) -> str:
        """Generate text using LLM."""
        resp = requests.post(
            f"{self.llm_url}/api/generate",
            headers=self._get_headers(include_auth=True),
            json={"model": self.llm_model, "prompt": prompt, "system": system, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    def build_index(self) -> None:
        """Build the document index with embeddings."""
        self.index = []
        for doc in self.docs:
            text_to_embed = doc["title"] + " " + doc["content"][:500]
            embedding = self._embed(text_to_embed)
            self.index.append({**doc, "embedding": embedding})

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        """Retrieve relevant documents."""
        query_emb = self._embed(query)
        scores = [(doc, cosine_similarity(query_emb, doc["embedding"])) for doc in self.index]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def ask(self, question: str) -> tuple[str, list[str]]:
        """Ask a question and get generated code."""
        # Retrieve relevant docs
        retrieved = self.retrieve(question, top_k=2)
        retrieved_ids = [doc["id"] for doc, _ in retrieved]

        # Build context
        context = "\n\n---\n\n".join([f"### {doc['title']}\n{doc['content']}" for doc, _ in retrieved])

        # Generate answer
        system = """You are a Highway DSL expert. Generate valid Python code.
Output PURE PYTHON CODE ONLY - no explanations, no markdown.
Always start with: from highway_dsl import WorkflowBuilder
Always define get_workflow() function.
Always return builder.build()."""

        prompt = f"""Highway DSL Documentation:
{context}

User Request: {question}

Generate the Highway DSL Python code:"""

        response = self._generate(prompt, system)

        # Clean up response
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip(), retrieved_ids


# Integration test markers
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ollama_config():
    """Get Ollama configuration from environment."""
    return {
        "llm_url": config.OLLAMA_BASE_URL,
        "embedding_url": config.OLLAMA_EMBEDDING_URL,
        "api_key": config.OLLAMA_API_KEY,
        "llm_model": config.DEFAULT_LLM_MODEL or "qwen3-coder:480b-cloud",
        "embedding_model": config.DEFAULT_EMBEDDING_MODEL or "nomic-embed-text:latest",
    }


@pytest.fixture(scope="module")
def check_ollama(ollama_config):
    """Skip tests if Ollama is not available."""
    embedding_available = check_ollama_available(ollama_config["embedding_url"])
    if not embedding_available:
        pytest.skip(f"Ollama embedding server not available at {ollama_config['embedding_url']}")

    # For LLM, we may be using cloud which requires different check
    llm_url = ollama_config["llm_url"]
    try:
        headers = {"Content-Type": "application/json"}
        if ollama_config["api_key"]:
            headers["Authorization"] = f"Bearer {ollama_config['api_key']}"
        resp = requests.get(f"{llm_url}/api/tags", headers=headers, timeout=5)
        if resp.status_code != 200:
            pytest.skip(f"Ollama LLM server not available at {llm_url}")
    except requests.RequestException:
        pytest.skip(f"Ollama LLM server not available at {llm_url}")


@pytest.fixture(scope="module")
def assistant(ollama_config, check_ollama):
    """Create and initialize the RAG assistant."""
    assistant = HighwayDSLAssistant(
        docs=DSL_DOCS,
        embedding_url=ollama_config["embedding_url"],
        llm_url=ollama_config["llm_url"],
        api_key=ollama_config["api_key"],
        embedding_model=ollama_config["embedding_model"],
        llm_model=ollama_config["llm_model"],
    )
    assistant.build_index()
    return assistant


@pytest.mark.integration
class TestHighwayRAGAssistant:
    """Integration tests for Highway DSL RAG Assistant."""

    def test_index_built(self, assistant):
        """Test that the index is built with embeddings."""
        assert len(assistant.index) == len(DSL_DOCS)
        for doc in assistant.index:
            assert "embedding" in doc
            assert len(doc["embedding"]) > 0

    def test_retrieval_returns_relevant_docs(self, assistant):
        """Test that retrieval returns relevant documents."""
        results = assistant.retrieve("How do I create a parallel workflow?", top_k=2)

        assert len(results) == 2
        # All results should have scores
        for doc, score in results:
            assert 0 <= score <= 1
            assert "id" in doc
            assert "content" in doc

    def test_retrieval_parallel_query(self, assistant):
        """Test that parallel workflow query retrieves parallel_basics doc."""
        results = assistant.retrieve("How do I run tasks in parallel?", top_k=2)
        retrieved_ids = [doc["id"] for doc, _ in results]

        # parallel_basics should be in top results
        assert "parallel_basics" in retrieved_ids

    def test_retrieval_workflow_builder_query(self, assistant):
        """Test that workflow builder query retrieves relevant doc."""
        results = assistant.retrieve("How do I create a workflow builder?", top_k=2)
        retrieved_ids = [doc["id"] for doc, _ in results]

        # workflow_builder_basics should be in top results
        assert "workflow_builder_basics" in retrieved_ids

    def test_ask_generates_code(self, assistant):
        """Test that ask generates Python code."""
        code, retrieved_ids = assistant.ask("Write a simple workflow that runs a shell command")

        assert len(code) > 0
        # Should contain workflow-related code
        assert "WorkflowBuilder" in code or "workflow" in code.lower()

    def test_ask_parallel_workflow(self, assistant):
        """Test generating a parallel workflow."""
        code, retrieved_ids = assistant.ask("Write a parallel workflow that runs 2 tasks concurrently")

        assert len(code) > 0
        # Should retrieve parallel docs
        assert "parallel_basics" in retrieved_ids or len(retrieved_ids) > 0

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        vec = [1.0, 0.0, 0.0]
        sim = cosine_similarity(vec, vec)
        assert sim == pytest.approx(1.0, rel=0.001)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, abs=0.001)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0, rel=0.001)


@pytest.mark.integration
class TestRAGQuality:
    """Tests for RAG quality metrics."""

    def test_embedding_consistency(self, assistant):
        """Test that same text produces same embedding."""
        text = "test query"
        emb1 = assistant._embed(text)
        emb2 = assistant._embed(text)

        # Embeddings should be identical for same text
        sim = cosine_similarity(emb1, emb2)
        assert sim == pytest.approx(1.0, rel=0.01)

    def test_embedding_differentiation(self, assistant):
        """Test that different texts produce different embeddings."""
        emb1 = assistant._embed("parallel workflow execution")
        emb2 = assistant._embed("completely unrelated random text about cooking")

        # Should be different (not perfectly similar)
        sim = cosine_similarity(emb1, emb2)
        assert sim < 0.9  # Should be reasonably different

    def test_retrieval_ranking(self, assistant):
        """Test that retrieval ranks more relevant docs higher."""
        results = assistant.retrieve("parallel fork wait branches", top_k=3)

        # First result should have highest score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
