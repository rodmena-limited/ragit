#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.config module.
"""

import os
import sys
from unittest.mock import patch


def reload_config_module():
    """Helper to reload config module with fresh environment."""
    # Remove cached modules to force fresh reload
    mods_to_remove = [k for k in sys.modules if k.startswith("ragit")]
    for mod in mods_to_remove:
        del sys.modules[mod]

    import ragit.config as config_module

    return config_module


class TestConfig:
    """Tests for Config class."""

    def test_config_default_values(self):
        """Test that Config has sensible defaults when env vars not set."""
        # Mock load_dotenv to prevent loading .env file
        with patch("dotenv.load_dotenv"):
            # Clear ALL relevant env vars
            env_vars_to_clear = [
                "OLLAMA_BASE_URL",
                "OLLAMA_API_KEY",
                "OLLAMA_TIMEOUT",
                "OLLAMA_EMBEDDING_URL",
                "RAGIT_DEFAULT_LLM_MODEL",
                "RAGIT_DEFAULT_EMBEDDING_MODEL",
                "RAGIT_LOG_LEVEL",
            ]
            clean_env = {k: v for k, v in os.environ.items() if k not in env_vars_to_clear}

            with patch.dict(os.environ, clean_env, clear=True):
                config_module = reload_config_module()
                cfg = config_module.Config()

                # Check defaults
                assert cfg.OLLAMA_BASE_URL == "http://localhost:11434"
                assert cfg.OLLAMA_API_KEY is None
                assert cfg.OLLAMA_TIMEOUT == 120
                assert cfg.LOG_LEVEL == "INFO"
                assert cfg.DEFAULT_LLM_MODEL == "qwen3-vl:235b-instruct"
                assert cfg.DEFAULT_EMBEDDING_MODEL == "nomic-embed-text:latest"

    def test_config_from_env_vars(self):
        """Test that Config loads values from environment variables."""
        test_env = {
            "OLLAMA_BASE_URL": "http://custom:8080",
            "OLLAMA_API_KEY": "my-secret-key",
            "OLLAMA_TIMEOUT": "60",
            "OLLAMA_EMBEDDING_URL": "http://embed:9090",
            "RAGIT_DEFAULT_LLM_MODEL": "custom-llm",
            "RAGIT_DEFAULT_EMBEDDING_MODEL": "custom-embed",
            "RAGIT_LOG_LEVEL": "DEBUG",
        }

        with patch.dict(os.environ, test_env, clear=True):
            config_module = reload_config_module()
            cfg = config_module.Config()

            assert cfg.OLLAMA_BASE_URL == "http://custom:8080"
            assert cfg.OLLAMA_API_KEY == "my-secret-key"
            assert cfg.OLLAMA_TIMEOUT == 60
            assert cfg.OLLAMA_EMBEDDING_URL == "http://embed:9090"
            assert cfg.DEFAULT_LLM_MODEL == "custom-llm"
            assert cfg.DEFAULT_EMBEDDING_MODEL == "custom-embed"
            assert cfg.LOG_LEVEL == "DEBUG"

    def test_config_embedding_url_fallback(self):
        """Test that embedding URL falls back to base URL if not set."""
        # Mock load_dotenv to prevent loading .env file
        with patch("dotenv.load_dotenv"):
            test_env = {
                "OLLAMA_BASE_URL": "http://fallback:11434",
                # OLLAMA_EMBEDDING_URL not set
            }

            with patch.dict(os.environ, test_env, clear=True):
                config_module = reload_config_module()
                cfg = config_module.Config()

                # Should fall back to base URL
                assert cfg.OLLAMA_EMBEDDING_URL == "http://fallback:11434"

    def test_config_singleton_instance(self):
        """Test that config singleton is created."""
        from ragit.config import config

        assert config is not None
        assert hasattr(config, "OLLAMA_BASE_URL")
        assert hasattr(config, "DEFAULT_LLM_MODEL")

    def test_config_timeout_integer_conversion(self):
        """Test that timeout is converted to integer."""
        test_env = {"OLLAMA_TIMEOUT": "45"}

        with patch.dict(os.environ, test_env, clear=True):
            config_module = reload_config_module()
            cfg = config_module.Config()

            assert isinstance(cfg.OLLAMA_TIMEOUT, int)
            assert cfg.OLLAMA_TIMEOUT == 45

    def test_config_all_attributes_exist(self):
        """Test that all config attributes exist."""
        from ragit.config import Config

        cfg = Config()

        # All attributes should exist
        assert hasattr(cfg, "OLLAMA_BASE_URL")
        assert hasattr(cfg, "OLLAMA_API_KEY")
        assert hasattr(cfg, "OLLAMA_TIMEOUT")
        assert hasattr(cfg, "OLLAMA_EMBEDDING_URL")
        assert hasattr(cfg, "DEFAULT_LLM_MODEL")
        assert hasattr(cfg, "DEFAULT_EMBEDDING_MODEL")
        assert hasattr(cfg, "LOG_LEVEL")
