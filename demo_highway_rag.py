#!/usr/bin/env python3
"""
RAG Demo: Highway Workflow Engine DSL Assistant
================================================

This demo uses RAG to help developers write Highway DSL workflow code
by retrieving relevant documentation and generating accurate code.
"""

import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434"
LLM = "qwen3-vl:235b-instruct-cloud"
EMBEDDER = "nomic-embed-text:latest"

# =============================================================================
# HIGHWAY DSL DOCUMENTATION (chunked for RAG)
# =============================================================================

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

If you forget the wait task, your workflow will continue immediately while branches are still running!"""
    },
    {
        "id": "parallel_branches_vs_tasks",
        "title": "Branches vs Tasks in Parallel",
        "content": """CRITICAL: Understanding Parallel Execution:

Number of BRANCHES = Number of CONCURRENT executions
Number of TASKS per branch = SEQUENTIAL chain within that branch

- Different BRANCHES run in PARALLEL - branch_1 and branch_2 execute concurrently
- Tasks WITHIN a branch run SEQUENTIALLY - .task().task().task() creates a chain

Example - 3 independent tasks running concurrently (3 branches, 1 task each):
```python
builder.parallel("fork", result_key="fork_data", branches={
    "branch_1": lambda b: b.task("task_1", "tools.shell.run", args=["echo '1'"]),
    "branch_2": lambda b: b.task("task_2", "tools.shell.run", args=["echo '2'"]),
    "branch_3": lambda b: b.task("task_3", "tools.shell.run", args=["echo '3'"]),
})
```

Example - 2 branches, each with 3 sequential tasks:
```python
builder.parallel("fork", result_key="fork_data", branches={
    "branch_1": lambda b: (
        b.task("b1_step1", "tools.shell.run", args=["echo 'Step 1'"])
         .task("b1_step2", "tools.shell.run", args=["echo 'Step 2'"])
         .task("b1_step3", "tools.shell.run", args=["echo 'Step 3'"])
    ),
    "branch_2": lambda b: (
        b.task("b2_step1", "tools.shell.run", args=["echo 'Step 1'"])
         .task("b2_step2", "tools.shell.run", args=["echo 'Step 2'"])
    ),
})
```"""
    },
    {
        "id": "complete_parallel_example",
        "title": "Complete Parallel Workflow Example",
        "content": """Complete parallel workflow with sleep:

```python
from highway_dsl import WorkflowBuilder

def get_workflow():
    builder = WorkflowBuilder(name="parallel_sleep_example")

    builder.parallel(
        "parallel_fork",
        result_key="fork_data",
        branches={
            "branch_1": lambda b: b.task(
                "sleep_branch_1",
                "tools.shell.run",
                args=["echo 'Branch 1 started' && sleep 10 && echo 'Branch 1 done'"],
                result_key="branch1_result"
            ),
            "branch_2": lambda b: b.task(
                "sleep_branch_2",
                "tools.shell.run",
                args=["echo 'Branch 2 started' && sleep 10 && echo 'Branch 2 done'"],
                result_key="branch2_result"
            ),
            "branch_3": lambda b: b.task(
                "sleep_branch_3",
                "tools.shell.run",
                args=["echo 'Branch 3 started' && sleep 10 && echo 'Branch 3 done'"],
                result_key="branch3_result"
            ),
        },
    )

    builder.task(
        "wait_for_branches",
        "tools.workflow.wait_for_parallel_branches",
        args=["{{fork_data}}"],
        kwargs={"timeout_seconds": 300},
        dependencies=["parallel_fork"],
    )

    return builder.build()
```

USE THIS EXACT PATTERN FOR ALL PARALLEL WORKFLOWS!"""
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
```"""
    },
    {
        "id": "variable_interpolation",
        "title": "Variable Interpolation",
        "content": """Access task results using template syntax:

- {{task_id.stdout}} - Shell command stdout
- {{task_id.stderr}} - Shell command stderr
- {{task_id.returncode}} - Shell command exit code
- {{task_id}} - Full task output
- {{task_id.response}} - HTTP response body
- {{item}} - Current item in foreach loop

CRITICAL: Use result_key + {{variable}} interpolation to pass data between tasks:

WRONG (filesystem workaround):
```python
builder.task("get_url", "tools.shell.run",
    args=["echo 'https://example.com' > /tmp/url.txt"])
builder.task("use_url", "tools.shell.run",
    args=["curl $(cat /tmp/url.txt)"])  # WRONG!
```

CORRECT (variable interpolation):
```python
builder.task("get_url", "tools.shell.run",
    args=["echo 'https://example.com'"],
    result_key="url_result")

builder.task("use_url", "tools.shell.run",
    args=["curl {{url_result.stdout}}"],  # CORRECT!
    dependencies=["get_url"])
```"""
    },
    {
        "id": "available_tools",
        "title": "Available Tool Functions",
        "content": """Core Tools:
- tools.shell.run - Execute shell commands
- tools.http.request - HTTP requests (GET, POST, PUT, DELETE)
- tools.python.run - Execute Python code with DurableContext

Communication Tools:
- tools.email.send - Send email (to, subject, body)
- tools.approval.request - Request human approval

Advanced Tools:
- tools.llm.call - Call LLM models (REQUIRES provider and model)
- tools.secrets.get_secret - HashiCorp Vault secret retrieval
- tools.workflow.wait_for_parallel_branches - Wait for parallel branches
- tools.workflow.execute - Execute nested workflows

Docker Tools:
- tools.docker.run - Run Docker container
- tools.docker.stop - Stop container
- tools.docker.compose_up - Start Docker Compose stack"""
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
```"""
    },
    {
        "id": "llm_tool",
        "title": "LLM Tool Integration",
        "content": """tools.llm.call - Call LLM models with circuit breaker protection.

CRITICAL: provider and model are REQUIRED - there are NO defaults!

Required Parameters:
- provider: "ollama", "openai", "anthropic", "grok", "gemini", "qwen"
- model: Model ID (e.g., "qwen3-vl:235b-instruct-cloud", "gpt-4o")
- prompt: The user prompt to send

Optional Parameters:
- system_prompt: System guidance
- temperature: 0.0-1.0 (default 0.7)
- max_tokens: Maximum tokens

Example:
```python
builder.task(
    "ask_llm",
    "tools.llm.call",
    kwargs={
        "provider": "ollama",
        "model": "qwen3-vl:235b-instruct-cloud",
        "prompt": "Explain workflow engines.",
        "temperature": 0.7,
    },
    result_key="llm_response",
)

builder.task(
    "use_response",
    "tools.shell.run",
    args=["echo 'LLM said: {{llm_response.response}}'"],
    dependencies=["ask_llm"],
)
```"""
    },
    {
        "id": "nested_parallelism",
        "title": "Nested Parallelism",
        "content": """When you need N branches, each with M parallel tasks - use NESTED parallel operators.

Example - 2 branches, each with 25 parallel tasks:
```python
builder.parallel("outer_fork", result_key="outer_fork_data", branches={
    "branch_1": lambda b: b.parallel("branch_1_fork", result_key="b1_fork_data", branches={
        **{f"b1_task_{i}": lambda b2, i=i: b2.task(
            f"write_b1_{i}",
            "tools.shell.run",
            args=[f"echo 'Branch 1 Task {i}'"]
        ) for i in range(1, 26)}
    }).task("b1_wait", "tools.workflow.wait_for_parallel_branches",
        args=["{{b1_fork_data}}"], dependencies=["branch_1_fork"]),

    "branch_2": lambda b: b.parallel("branch_2_fork", result_key="b2_fork_data", branches={
        **{f"b2_task_{i}": lambda b2, i=i: b2.task(
            f"write_b2_{i}",
            "tools.shell.run",
            args=[f"echo 'Branch 2 Task {i}'"]
        ) for i in range(1, 26)}
    }).task("b2_wait", "tools.workflow.wait_for_parallel_branches",
        args=["{{b2_fork_data}}"], dependencies=["branch_2_fork"]),
})

# Wait for outer branches
builder.task("wait_all", "tools.workflow.wait_for_parallel_branches",
    args=["{{outer_fork_data}}"], dependencies=["outer_fork"])
```"""
    },
]

# =============================================================================
# RAG SYSTEM
# =============================================================================

def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBEDDER, "input": text}, timeout=60)
    return np.array(r.json()["embeddings"][0])

def generate(prompt, system):
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": LLM, "prompt": prompt, "system": system, "stream": False},
                      timeout=180)
    return r.json()["response"]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class HighwayDSLAssistant:
    """RAG-powered Highway DSL code assistant"""

    def __init__(self, docs):
        print("📚 Indexing Highway DSL documentation...")
        self.index = []
        for doc in docs:
            emb = embed(doc["title"] + " " + doc["content"][:500])
            self.index.append({**doc, "embedding": emb})
        print(f"✅ Indexed {len(self.index)} documentation chunks\n")

    def retrieve(self, query, top_k=3):
        q_emb = embed(query)
        scores = [(doc, cosine_sim(q_emb, doc["embedding"])) for doc in self.index]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def ask(self, question):
        print(f"\n📝 Query: {question}")
        print("-" * 60)

        # Retrieve relevant docs
        retrieved = self.retrieve(question, top_k=3)
        print(f"📄 Retrieved: {[d[0]['id'] for d in retrieved]}")

        # Build context
        context = "\n\n---\n\n".join([
            f"### {doc['title']}\n{doc['content']}"
            for doc, _ in retrieved
        ])

        # Generate answer
        system = """You are a Highway DSL expert assistant. Generate ONLY valid Python code using the highway_dsl package.

RULES:
1. Output PURE PYTHON CODE ONLY - no explanations, no markdown
2. Always start with: from highway_dsl import WorkflowBuilder
3. Always define get_workflow() function
4. Always return builder.build()
5. For parallel workflows, ALWAYS include wait_for_parallel_branches task
6. Use the exact patterns from the documentation"""

        prompt = f"""Highway DSL Documentation:
{context}

User Request: {question}

Generate the Highway DSL Python code:"""

        print("\n🤖 Generating code...\n")
        response = generate(prompt, system)

        # Clean up response (remove markdown if present)
        code = response
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   RAG Demo: Highway DSL Code Assistant                                ║
║                                                                       ║
║   This assistant uses RAG to retrieve relevant DSL documentation      ║
║   and generate correct Highway workflow code.                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Verify connection
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        print(f"✓ Ollama connected | LLM: {LLM}")
    except:
        print("✗ Cannot connect to Ollama")
        return

    # Initialize RAG assistant
    assistant = HighwayDSLAssistant(DSL_DOCS)

    # Test query: Write parallel workflow code
    query = "Write a parallel workflow that runs 3 tasks concurrently: one downloads a file, one processes data, and one sends an email notification. Wait for all to complete."

    code = assistant.ask(query)

    print("=" * 60)
    print("GENERATED CODE:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    print("""
HOW RAG HELPED:
---------------
1. Retrieved relevant documentation about:
   - Parallel workflow patterns
   - The required wait_for_parallel_branches task
   - Correct branch syntax

2. Generated code that follows Highway DSL patterns:
   - Uses WorkflowBuilder correctly
   - Includes the REQUIRED wait task (most common mistake!)
   - Uses proper result_key for data passing

Without RAG, the LLM might generate:
   - Prefect/Airflow code (wrong framework!)
   - Missing the wait task (workflow would fail!)
   - Wrong syntax for branches
""")


if __name__ == "__main__":
    main()
