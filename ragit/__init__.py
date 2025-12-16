#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit - RAG toolkit for document Q&A and hyperparameter optimization.

Quick Start
-----------
>>> from ragit import RAGAssistant
>>>
>>> # Load docs and ask questions
>>> assistant = RAGAssistant("docs/")
>>> answer = assistant.ask("How do I create a REST API?")
>>> print(answer)
>>>
>>> # Generate code
>>> code = assistant.generate_code("create a user authentication API")
>>> print(code)

Optimization
------------
>>> from ragit import RagitExperiment, Document, BenchmarkQuestion
>>>
>>> docs = [Document(id="doc1", content="...")]
>>> benchmark = [BenchmarkQuestion(question="What is X?", ground_truth="...")]
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

# Public API (imports after logging setup)
from ragit.assistant import RAGAssistant  # noqa: E402
from ragit.core.experiment.experiment import (  # noqa: E402
    BenchmarkQuestion,
    Chunk,
    Document,
    RAGConfig,
    RagitExperiment,
)
from ragit.core.experiment.results import EvaluationResult, ExperimentResults  # noqa: E402
from ragit.loaders import (  # noqa: E402
    chunk_by_separator,
    chunk_document,
    chunk_rst_sections,
    chunk_text,
    load_directory,
    load_text,
)
from ragit.providers import OllamaProvider  # noqa: E402

__all__ = [
    "__version__",
    # High-level API
    "RAGAssistant",
    # Document loading
    "load_text",
    "load_directory",
    "chunk_text",
    "chunk_document",
    "chunk_by_separator",
    "chunk_rst_sections",
    # Core classes
    "Document",
    "Chunk",
    "OllamaProvider",
    # Optimization
    "RagitExperiment",
    "BenchmarkQuestion",
    "RAGConfig",
    "EvaluationResult",
    "ExperimentResults",
]
