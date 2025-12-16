#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit - Automatic RAG Pattern Optimization Engine

A tool for automatically finding optimal hyperparameters for RAG
(Retrieval-Augmented Generation) pipelines.

Supports:
- Ollama (local LLM and embeddings)
- Future: Gemini, Claude, OpenAI

Example
-------
>>> from ragit import RagitExperiment, Document, BenchmarkQuestion, OllamaProvider
>>>
>>> docs = [Document(id="doc1", content="Machine learning is...")]
>>> benchmark = [BenchmarkQuestion(question="What is ML?", ground_truth="...")]
>>>
>>> experiment = RagitExperiment(docs, benchmark)
>>> results = experiment.run()
>>> print(results[0])  # Best configuration
"""

import logging
import os

from ragit.version import __version__

# Set up logging
logger = logging.getLogger("ragit")
logger.setLevel(os.getenv("RAGIT_LOG_LEVEL", "INFO"))

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Public API
from ragit.providers import OllamaProvider
from ragit.core.experiment.experiment import (
    RagitExperiment,
    Document,
    BenchmarkQuestion,
    RAGConfig,
)
from ragit.core.experiment.results import EvaluationResult, ExperimentResults

__all__ = [
    "__version__",
    "RagitExperiment",
    "Document",
    "BenchmarkQuestion",
    "RAGConfig",
    "EvaluationResult",
    "ExperimentResults",
    "OllamaProvider",
]
