#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit configuration management.

Loads configuration from environment variables and .env files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from current working directory or project root
_env_path = Path.cwd() / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Try to find .env in parent directories
    for parent in Path.cwd().parents:
        _env_path = parent / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
            break


class Config:
    """Ragit configuration loaded from environment variables."""

    # Ollama LLM API Configuration (can be cloud)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_API_KEY: str | None = os.getenv("OLLAMA_API_KEY")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Ollama Embedding API Configuration (cloud doesn't support embeddings, use local)
    OLLAMA_EMBEDDING_URL: str = os.getenv(
        "OLLAMA_EMBEDDING_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Default Models
    DEFAULT_LLM_MODEL: str = os.getenv("RAGIT_DEFAULT_LLM_MODEL", "qwen3-vl:235b-instruct")
    DEFAULT_EMBEDDING_MODEL: str = os.getenv("RAGIT_DEFAULT_EMBEDDING_MODEL", "mxbai-embed-large")

    # Logging
    LOG_LEVEL: str = os.getenv("RAGIT_LOG_LEVEL", "INFO")


# Singleton instance
config = Config()
