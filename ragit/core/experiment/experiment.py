#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit Experiment - Core RAG optimization engine.

This module provides the main experiment class for optimizing RAG hyperparameters.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from ragit.core.experiment.results import EvaluationResult
from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from ragit.providers.function_adapter import FunctionProvider


@dataclass
class RAGConfig:
    """Configuration for a RAG pattern."""

    name: str
    chunk_size: int
    chunk_overlap: int
    num_chunks: int  # Number of chunks to retrieve
    embedding_model: str
    llm_model: str


@dataclass
class Document:
    """A document in the knowledge base."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A document chunk."""

    content: str
    doc_id: str
    chunk_index: int
    embedding: tuple[float, ...] | list[float] | None = None


@dataclass
class BenchmarkQuestion:
    """A benchmark question for evaluation."""

    question: str
    ground_truth: str
    relevant_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationScores:
    """Scores from evaluating a RAG response."""

    answer_correctness: float
    context_relevance: float
    faithfulness: float

    @property
    def combined_score(self) -> float:
        """Combined score (weighted average)."""
        return 0.4 * self.answer_correctness + 0.3 * self.context_relevance + 0.3 * self.faithfulness


class SimpleVectorStore:
    """Simple in-memory vector store with pre-normalized embeddings for fast search.

    Note: This class is NOT thread-safe.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._embedding_matrix: np.ndarray[Any, np.dtype[np.float64]] | None = None  # Pre-normalized

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store and rebuild pre-normalized embedding matrix."""
        self.chunks.extend(chunks)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuild and pre-normalize the embedding matrix from chunks."""
        embeddings = [c.embedding for c in self.chunks if c.embedding is not None]
        if embeddings:
            matrix = np.array(embeddings, dtype=np.float64)
            # Pre-normalize for fast cosine similarity
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            self._embedding_matrix = matrix / norms
        else:
            self._embedding_matrix = None

    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks = []
        self._embedding_matrix = None

    def search(self, query_embedding: tuple[float, ...] | list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search for similar chunks using pre-normalized cosine similarity."""
        if not self.chunks or self._embedding_matrix is None:
            return []

        # Normalize query vector
        query_vec = np.array(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normalized = query_vec / query_norm

        # Fast cosine similarity: matrix is pre-normalized, just dot product
        similarities = self._embedding_matrix @ query_normalized

        # Get top_k indices efficiently
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]


class RagitExperiment:
    """
    Ragit Experiment - Automatic RAG Hyperparameter Optimization.

    This class orchestrates the optimization of RAG pipeline hyperparameters
    by systematically evaluating different configurations.

    Parameters
    ----------
    documents : list[Document]
        Documents to use as the knowledge base.
    benchmark : list[BenchmarkQuestion]
        Benchmark questions for evaluation.
    embed_fn : Callable[[str], list[float]], optional
        Function that takes text and returns an embedding vector.
    generate_fn : Callable, optional
        Function for text generation.
    provider : BaseEmbeddingProvider, optional
        Provider for embeddings and LLM. If embed_fn is provided, this is
        ignored for embeddings but can be used for LLM.

    Raises
    ------
    ValueError
        If neither embed_fn nor provider is provided.

    Examples
    --------
    >>> # With custom functions
    >>> experiment = RagitExperiment(docs, benchmark, embed_fn=my_embed, generate_fn=my_llm)
    >>>
    >>> # With explicit provider
    >>> from ragit.providers import OllamaProvider
    >>> experiment = RagitExperiment(docs, benchmark, provider=OllamaProvider())
    >>>
    >>> results = experiment.run()
    >>> print(results[0].config)  # Best configuration
    """

    def __init__(
        self,
        documents: list[Document],
        benchmark: list[BenchmarkQuestion],
        embed_fn: Callable[[str], list[float]] | None = None,
        generate_fn: Callable[..., str] | None = None,
        provider: BaseEmbeddingProvider | BaseLLMProvider | None = None,
    ):
        self.documents = documents
        self.benchmark = benchmark
        self.vector_store = SimpleVectorStore()
        self.results: list[EvaluationResult] = []

        # Resolve provider from functions or explicit provider
        self._embedding_provider: BaseEmbeddingProvider
        self._llm_provider: BaseLLMProvider | None = None

        if embed_fn is not None:
            # Create FunctionProvider from provided functions
            function_provider = FunctionProvider(
                embed_fn=embed_fn,
                generate_fn=generate_fn,
            )
            self._embedding_provider = function_provider
            if generate_fn is not None:
                self._llm_provider = function_provider
            elif provider is not None and isinstance(provider, BaseLLMProvider):
                self._llm_provider = provider
        elif provider is not None:
            if not isinstance(provider, BaseEmbeddingProvider):
                raise ValueError(
                    "Provider must implement BaseEmbeddingProvider for embeddings. "
                    "Alternatively, provide embed_fn."
                )
            self._embedding_provider = provider
            if isinstance(provider, BaseLLMProvider):
                self._llm_provider = provider
        else:
            raise ValueError(
                "Must provide embed_fn or provider for embeddings. "
                "Examples:\n"
                "  RagitExperiment(docs, benchmark, embed_fn=my_embed, generate_fn=my_llm)\n"
                "  RagitExperiment(docs, benchmark, provider=OllamaProvider())"
            )

        # LLM is required for evaluation
        if self._llm_provider is None:
            raise ValueError(
                "RagitExperiment requires LLM for evaluation. "
                "Provide generate_fn or a provider with LLM support."
            )

    @property
    def provider(self) -> BaseEmbeddingProvider:
        """Return the embedding provider (for backwards compatibility)."""
        return self._embedding_provider

    def define_search_space(
        self,
        chunk_sizes: list[int] | None = None,
        chunk_overlaps: list[int] | None = None,
        num_chunks_options: list[int] | None = None,
        embedding_models: list[str] | None = None,
        llm_models: list[str] | None = None,
    ) -> list[RAGConfig]:
        """
        Define the hyperparameter search space.

        Parameters
        ----------
        chunk_sizes : list[int], optional
            Chunk sizes to test. Default: [256, 512]
        chunk_overlaps : list[int], optional
            Chunk overlaps to test. Default: [50, 100]
        num_chunks_options : list[int], optional
            Number of chunks to retrieve. Default: [2, 3]
        embedding_models : list[str], optional
            Embedding models to test. Default: ["default"]
        llm_models : list[str], optional
            LLM models to test. Default: ["default"]

        Returns
        -------
        list[RAGConfig]
            List of configurations to evaluate.
        """
        chunk_sizes = chunk_sizes or [256, 512]
        chunk_overlaps = chunk_overlaps or [50, 100]
        num_chunks_options = num_chunks_options or [2, 3]
        embedding_models = embedding_models or ["default"]
        llm_models = llm_models or ["default"]

        configs = []
        pattern_num = 1

        for cs, co, nc, em, lm in product(
            chunk_sizes, chunk_overlaps, num_chunks_options, embedding_models, llm_models
        ):
            # Ensure overlap is less than chunk size
            if co >= cs:
                continue

            configs.append(
                RAGConfig(
                    name=f"Pattern_{pattern_num}",
                    chunk_size=cs,
                    chunk_overlap=co,
                    num_chunks=nc,
                    embedding_model=em,
                    llm_model=lm,
                )
            )
            pattern_num += 1

        return configs

    def _chunk_document(self, doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
        """Split document into overlapping chunks."""
        chunks = []
        text = doc.content
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        doc_id=doc.id,
                        chunk_index=chunk_idx,
                    )
                )
                chunk_idx += 1

            start = end - overlap
            if start >= len(text) - overlap:
                break

        return chunks

    def _build_index(self, config: RAGConfig) -> None:
        """Build vector index with given configuration using batch embedding."""
        self.vector_store.clear()
        all_chunks: list[Chunk] = []

        # Chunk all documents
        for doc in self.documents:
            chunks = self._chunk_document(doc, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Batch embed all chunks at once (single API call)
        texts = [chunk.content for chunk in all_chunks]
        responses = self._embedding_provider.embed_batch(texts, config.embedding_model)

        for chunk, response in zip(all_chunks, responses, strict=True):
            chunk.embedding = response.embedding

        self.vector_store.add(all_chunks)

    def _retrieve(self, query: str, config: RAGConfig) -> list[Chunk]:
        """Retrieve relevant chunks for a query."""
        query_response = self._embedding_provider.embed(query, config.embedding_model)
        results = self.vector_store.search(query_response.embedding, top_k=config.num_chunks)
        return [chunk for chunk, _ in results]

    def _generate(self, question: str, context: str, config: RAGConfig) -> str:
        """Generate answer using RAG."""
        if self._llm_provider is None:
            raise ValueError("LLM provider is required for generation")

        system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so. Be concise and accurate."""

        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        response = self._llm_provider.generate(
            prompt=prompt,
            model=config.llm_model,
            system_prompt=system_prompt,
            temperature=0.7,
        )
        return response.text

    def _evaluate_response(
        self,
        question: str,
        generated: str,
        ground_truth: str,
        context: str,
        config: RAGConfig,
    ) -> EvaluationScores:
        """Evaluate a RAG response using LLM-as-judge."""
        if self._llm_provider is None:
            raise ValueError("LLM provider is required for evaluation")

        def extract_score(response: str) -> float:
            """Extract numeric score from LLM response."""
            try:
                # Find first number in response
                nums = "".join(c for c in response if c.isdigit() or c == ".")
                if nums:
                    score = float(nums.split(".")[0])  # Take integer part
                    return min(100, max(0, score)) / 100
            except (ValueError, IndexError):
                pass
            return 0.5

        # Evaluate answer correctness
        correctness_prompt = f"""Rate how correct this answer is compared to ground truth (0-100):

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated}

Respond with ONLY a number 0-100."""

        resp = self._llm_provider.generate(correctness_prompt, config.llm_model)
        correctness = extract_score(resp.text)

        # Evaluate context relevance
        relevance_prompt = f"""Rate how relevant this context is for answering the question (0-100):

Question: {question}
Context: {context[:1000]}

Respond with ONLY a number 0-100."""

        resp = self._llm_provider.generate(relevance_prompt, config.llm_model)
        relevance = extract_score(resp.text)

        # Evaluate faithfulness
        faithfulness_prompt = f"""Rate if this answer is grounded in the context (0-100):

Context: {context[:1000]}
Answer: {generated}

Respond with ONLY a number 0-100."""

        resp = self._llm_provider.generate(faithfulness_prompt, config.llm_model)
        faithfulness = extract_score(resp.text)

        return EvaluationScores(
            answer_correctness=correctness,
            context_relevance=relevance,
            faithfulness=faithfulness,
        )

    def evaluate_config(self, config: RAGConfig, verbose: bool = False) -> EvaluationResult:
        """
        Evaluate a single RAG configuration.

        Parameters
        ----------
        config : RAGConfig
            Configuration to evaluate.
        verbose : bool
            Print progress information.

        Returns
        -------
        EvaluationResult
            Evaluation results for this configuration.
        """
        if verbose:
            print(f"\nEvaluating {config.name}:")
            print(f"  chunk_size={config.chunk_size}, overlap={config.chunk_overlap}, num_chunks={config.num_chunks}")

        start_time = time.time()

        # Build index
        self._build_index(config)

        # Evaluate on benchmark
        all_scores = []

        for qa in self.benchmark:
            # Retrieve
            chunks = self._retrieve(qa.question, config)
            context = "\n\n".join(f"[{c.doc_id}]: {c.content}" for c in chunks)

            # Generate
            answer = self._generate(qa.question, context, config)

            # Evaluate
            scores = self._evaluate_response(qa.question, answer, qa.ground_truth, context, config)
            all_scores.append(scores)

        # Aggregate scores (use generators for memory efficiency)
        avg_correctness = np.mean([s.answer_correctness for s in all_scores])
        avg_relevance = np.mean([s.context_relevance for s in all_scores])
        avg_faithfulness = np.mean([s.faithfulness for s in all_scores])
        combined = float(np.mean([s.combined_score for s in all_scores]))

        execution_time = time.time() - start_time

        if verbose:
            print(
                f"  Scores: correctness={avg_correctness:.2f}, "
                f"relevance={avg_relevance:.2f}, faithfulness={avg_faithfulness:.2f}"
            )
            print(f"  Combined: {combined:.3f} | Time: {execution_time:.1f}s")

        return EvaluationResult(
            pattern_name=config.name,
            indexing_params={
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "embedding_model": config.embedding_model,
            },
            inference_params={
                "num_chunks": config.num_chunks,
                "llm_model": config.llm_model,
            },
            scores={
                "answer_correctness": {"mean": float(avg_correctness)},
                "context_relevance": {"mean": float(avg_relevance)},
                "faithfulness": {"mean": float(avg_faithfulness)},
            },
            execution_time=execution_time,
            final_score=float(combined),
        )

    def run(
        self,
        configs: list[RAGConfig] | None = None,
        max_configs: int | None = None,
        verbose: bool = True,
    ) -> list[EvaluationResult]:
        """
        Run the RAG optimization experiment.

        Parameters
        ----------
        configs : list[RAGConfig], optional
            Configurations to evaluate. If None, uses default search space.
        max_configs : int, optional
            Maximum number of configurations to evaluate.
        verbose : bool
            Print progress information.

        Returns
        -------
        list[EvaluationResult]
            Results sorted by combined score (best first).
        """
        if configs is None:
            configs = self.define_search_space()

        if max_configs:
            configs = configs[:max_configs]

        if verbose:
            print("=" * 60)
            print("RAGIT: RAG Optimization Experiment")
            print("=" * 60)
            print(f"Configurations to test: {len(configs)}")
            print(f"Documents: {len(self.documents)}")
            print(f"Benchmark questions: {len(self.benchmark)}")
            print()

        self.results = []

        for cfg in tqdm(configs, desc="Evaluating configs", disable=not verbose):
            result = self.evaluate_config(cfg, verbose=verbose)
            self.results.append(result)

        # Sort by combined score (best first)
        self.results.sort(key=lambda x: x.final_score, reverse=True)

        if verbose:
            print("\n" + "=" * 60)
            print("RESULTS (sorted by score)")
            print("=" * 60)
            for i, result in enumerate(self.results[:5], 1):
                print(f"{i}. {result.pattern_name}: {result.final_score:.3f}")
                print(
                    f"   chunk_size={result.indexing_params['chunk_size']}, "
                    f"num_chunks={result.inference_params['num_chunks']}"
                )

        return self.results

    def get_best_config(self) -> EvaluationResult | None:
        """Get the best configuration from results."""
        if not self.results:
            return None
        return self.results[0]
