#!/usr/bin/env python3
"""
RAG Demo: Generate Falcon Web App Code

Uses ragit to generate correct Falcon framework code from documentation.

Usage:
    python demos/build-falcon-app-example.py
    python demos/build-falcon-app-example.py "create a user API with login"
"""

import sys
from pathlib import Path

from ragit import RAGAssistant
from ragit.providers import OllamaProvider

# Falcon-specific system prompt
FALCON_SYSTEM_PROMPT = """You are a Falcon web framework expert. Generate ONLY valid Python code.

RULES:
1. Output PURE PYTHON CODE ONLY - no explanations, no markdown
2. Always import falcon and json
3. Use falcon.App() to create the application
4. Use on_get, on_post, on_put, on_delete methods for HTTP verbs
5. Use resp.text for JSON responses, resp.status for status codes
6. Use req.media to read JSON request body
7. Use falcon.HTTPNotFound(), falcon.HTTPBadRequest() for errors
8. Add routes with app.add_route('/path', ResourceClass())
9. Include a comment showing how to run: gunicorn app:app"""


def main():
    # Get request from args or use default
    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = "Create a REST API for a todo list with GET all, GET by id, POST, PUT, DELETE"

    # Load Falcon documentation and create assistant
    docs_path = Path(__file__).parent / "falcon-tutorial.rst"
    assistant = RAGAssistant(
        docs_path,
        provider=OllamaProvider(),
        embedding_model="nomic-embed-text:latest",
        llm_model="qwen3-coder:480b-cloud",
    )

    # Generate code with custom system prompt
    context = assistant.get_context(request, top_k=3)
    prompt = f"""Falcon Framework Documentation:
{context}

Request: {request}

Generate complete, runnable Falcon Python code:"""

    code = assistant.generate(prompt, FALCON_SYSTEM_PROMPT)

    # Clean markdown if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    print(code.strip())


if __name__ == "__main__":
    main()
