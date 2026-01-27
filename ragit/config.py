#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit configuration management with Pydantic validation.

Loads configuration from environment variables and .env files.
Validates all configuration values at startup.

Note: As of v0.8.0, ragit no longer has default LLM or embedding models.
Users must explicitly configure providers.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Note: We define ConfigValidationError locally to avoid circular imports,
# but ragit.exceptions.ConfigurationError can be used elsewhere

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


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class RagitConfig(BaseModel):
    """Validated ragit configuration.

    All configuration values are validated at startup. Invalid values
    raise ConfigValidationError with a descriptive message.

    Attributes
    ----------
    ollama_base_url : str
        Ollama server URL (default: http://localhost:11434)
    ollama_embedding_url : str
        Embedding API URL (defaults to ollama_base_url)
    ollama_api_key : str | None
        API key for authentication
    ollama_timeout : int
        Request timeout in seconds (1-600)
    default_llm_model : str | None
        Default LLM model name
    default_embedding_model : str | None
        Default embedding model name
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_embedding_url: str | None = None
    ollama_api_key: str | None = None
    ollama_timeout: int = Field(default=120, gt=0, le=600)
    default_llm_model: str | None = None
    default_embedding_model: str | None = None
    log_level: str = Field(default="INFO")

    @field_validator("ollama_base_url", "ollama_embedding_url", mode="before")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format."""
        if v is None:
            return v
        v = str(v).strip().rstrip("/")
        if not v:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL must start with http:// or https://: {v}")
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = str(v).upper().strip()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v

    @field_validator("ollama_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Treat empty string as None."""
        if v is not None and not str(v).strip():
            return None
        return v

    @field_validator("ollama_timeout", mode="before")
    @classmethod
    def validate_timeout(cls, v: int | str) -> int:
        """Parse and validate timeout value."""
        try:
            timeout = int(v)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid timeout value '{v}': must be an integer") from e
        return timeout

    model_config = {"extra": "forbid"}

    # Uppercase aliases for backwards compatibility
    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self.ollama_base_url

    @property
    def OLLAMA_EMBEDDING_URL(self) -> str:
        return self.ollama_embedding_url or self.ollama_base_url

    @property
    def OLLAMA_API_KEY(self) -> str | None:
        return self.ollama_api_key

    @property
    def OLLAMA_TIMEOUT(self) -> int:
        return self.ollama_timeout

    @property
    def DEFAULT_LLM_MODEL(self) -> str | None:
        return self.default_llm_model

    @property
    def DEFAULT_EMBEDDING_MODEL(self) -> str | None:
        return self.default_embedding_model

    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level


def _safe_get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable, returning None for empty strings."""
    value = os.getenv(key, default)
    if value is not None and not value.strip():
        return default
    return value


def _safe_get_int_env(key: str, default: int) -> int | str:
    """Get environment variable as int, returning raw string if invalid."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        # Return the raw string so Pydantic can give a better error message
        return value


def load_config() -> RagitConfig:
    """Load and validate configuration from environment variables.

    Returns
    -------
    RagitConfig
        Validated configuration object.

    Raises
    ------
    ConfigValidationError
        If configuration validation fails.
    """
    try:
        return RagitConfig(
            ollama_base_url=_safe_get_env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
            ollama_embedding_url=_safe_get_env("OLLAMA_EMBEDDING_URL") or _safe_get_env("OLLAMA_BASE_URL"),
            ollama_api_key=_safe_get_env("OLLAMA_API_KEY"),
            ollama_timeout=_safe_get_int_env("OLLAMA_TIMEOUT", 120),
            default_llm_model=_safe_get_env("RAGIT_DEFAULT_LLM_MODEL"),
            default_embedding_model=_safe_get_env("RAGIT_DEFAULT_EMBEDDING_MODEL"),
            log_level=_safe_get_env("RAGIT_LOG_LEVEL", "INFO") or "INFO",
        )
    except Exception as e:
        raise ConfigValidationError(f"Configuration error: {e}") from e


# Singleton instance - validates configuration at import time
try:
    config = load_config()
except ConfigValidationError as e:
    # Re-raise with clear message
    raise ConfigValidationError(str(e)) from e


# Backwards compatibility alias
Config = RagitConfig
