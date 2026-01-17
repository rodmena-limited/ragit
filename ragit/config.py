#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit configuration management.

Loads configuration from environment variables and .env files.

Note: As of v0.8.0, ragit no longer has default LLM or embedding models.
Users must explicitly configure providers.
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
    """Ragit configuration loaded from environment variables.

    Note: As of v0.8.0, DEFAULT_LLM_MODEL and DEFAULT_EMBEDDING_MODEL are
    no longer used as defaults. They are only read from environment variables
    for backwards compatibility with user configurations.
    """

    # Ollama LLM API Configuration (used when explicitly using OllamaProvider)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_API_KEY: str | None = os.getenv("OLLAMA_API_KEY")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Ollama Embedding API Configuration
    OLLAMA_EMBEDDING_URL: str = os.getenv(
        "OLLAMA_EMBEDDING_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Model settings (only used if explicitly requested, no defaults)
    # These can still be set via environment variables for convenience
    DEFAULT_LLM_MODEL: str | None = os.getenv("RAGIT_DEFAULT_LLM_MODEL")
    DEFAULT_EMBEDDING_MODEL: str | None = os.getenv("RAGIT_DEFAULT_EMBEDDING_MODEL")

    # Logging
    LOG_LEVEL: str = os.getenv("RAGIT_LOG_LEVEL", "INFO")


# Singleton instance
config = Config()
