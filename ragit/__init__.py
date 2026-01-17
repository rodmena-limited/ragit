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
>>> # With custom embedding function (retrieval-only)
>>> def my_embed(text: str) -> list[float]:
...     # Your embedding implementation
...     pass
>>> assistant = RAGAssistant("docs/", embed_fn=my_embed)
>>> results = assistant.retrieve("How do I create a REST API?")
>>>
>>> # With SentenceTransformers (offline, requires ragit[transformers])
>>> from ragit.providers import SentenceTransformersProvider
>>> assistant = RAGAssistant("docs/", provider=SentenceTransformersProvider())
>>>
>>> # With Ollama (explicit)
>>> from ragit.providers import OllamaProvider
>>> assistant = RAGAssistant("docs/", provider=OllamaProvider())
>>> answer = assistant.ask("How do I create a REST API?")

Optimization
------------
>>> from ragit import RagitExperiment, Document, BenchmarkQuestion
>>>
>>> docs = [Document(id="doc1", content="...")]
>>> benchmark = [BenchmarkQuestion(question="What is X?", ground_truth="...")]
>>>
>>> # With explicit provider
>>> experiment = RagitExperiment(docs, benchmark, provider=OllamaProvider())
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
from ragit.providers import (  # noqa: E402
    BaseEmbeddingProvider,
    BaseLLMProvider,
    FunctionProvider,
    OllamaProvider,
)

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
    # Providers
    "OllamaProvider",
    "FunctionProvider",
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    # Optimization
    "RagitExperiment",
    "BenchmarkQuestion",
    "RAGConfig",
    "EvaluationResult",
    "ExperimentResults",
]

# Conditionally add SentenceTransformersProvider if available
try:
    from ragit.providers import (  # noqa: E402
        SentenceTransformersProvider as SentenceTransformersProvider,
    )

    __all__ += ["SentenceTransformersProvider"]
except ImportError:
    pass
