#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit Experiment - Core RAG optimization engine.

This module provides the main experiment class for optimizing RAG hyperparameters.
"""

import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from tqdm import tqdm

from ragit.config import config
from ragit.core.experiment.results import EvaluationResult
from ragit.providers import OllamaProvider


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
    embedding: list[float] | None = None


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
    """Simple in-memory vector store."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the store."""
        self.chunks.extend(chunks)

    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks = []

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search for similar chunks."""
        if not self.chunks:
            return []

        scores = []
        query = np.array(query_embedding)

        for chunk in self.chunks:
            if chunk.embedding:
                chunk_emb = np.array(chunk.embedding)
                # Cosine similarity
                similarity = np.dot(query, chunk_emb) / (np.linalg.norm(query) * np.linalg.norm(chunk_emb))
                scores.append((chunk, float(similarity)))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


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
    provider : OllamaProvider, optional
        LLM/Embedding provider. Defaults to OllamaProvider().

    Examples
    --------
    >>> documents = [Document(id="doc1", content="...")]
    >>> benchmark = [BenchmarkQuestion(question="...", ground_truth="...")]
    >>> experiment = RagitExperiment(documents, benchmark)
    >>> results = experiment.run()
    >>> print(results[0].config)  # Best configuration
    """

    def __init__(
        self,
        documents: list[Document],
        benchmark: list[BenchmarkQuestion],
        provider: OllamaProvider | None = None,
    ):
        self.documents = documents
        self.benchmark = benchmark
        self.provider = provider or OllamaProvider()
        self.vector_store = SimpleVectorStore()
        self.results: list[EvaluationResult] = []

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
            Number of chunks to retrieve. Default: [2, 3, 5]
        embedding_models : list[str], optional
            Embedding models to test. Default: from RAGIT_DEFAULT_EMBEDDING_MODEL env var
        llm_models : list[str], optional
            LLM models to test. Default: from RAGIT_DEFAULT_LLM_MODEL env var

        Returns
        -------
        list[RAGConfig]
            List of configurations to evaluate.
        """
        chunk_sizes = chunk_sizes or [256, 512]
        chunk_overlaps = chunk_overlaps or [50, 100]
        num_chunks_options = num_chunks_options or [2, 3]
        embedding_models = embedding_models or [config.DEFAULT_EMBEDDING_MODEL]
        llm_models = llm_models or [config.DEFAULT_LLM_MODEL]

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
        """Build vector index with given configuration."""
        self.vector_store.clear()
        all_chunks = []

        # Chunk all documents
        for doc in self.documents:
            chunks = self._chunk_document(doc, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)

        # Embed all chunks
        for chunk in all_chunks:
            response = self.provider.embed(chunk.content, config.embedding_model)
            chunk.embedding = response.embedding

        self.vector_store.add(all_chunks)

    def _retrieve(self, query: str, config: RAGConfig) -> list[Chunk]:
        """Retrieve relevant chunks for a query."""
        query_response = self.provider.embed(query, config.embedding_model)
        results = self.vector_store.search(query_response.embedding, top_k=config.num_chunks)
        return [chunk for chunk, _ in results]

    def _generate(self, question: str, context: str, config: RAGConfig) -> str:
        """Generate answer using RAG."""
        system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so. Be concise and accurate."""

        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        response = self.provider.generate(
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

        resp = self.provider.generate(correctness_prompt, config.llm_model)
        correctness = extract_score(resp.text)

        # Evaluate context relevance
        relevance_prompt = f"""Rate how relevant this context is for answering the question (0-100):

Question: {question}
Context: {context[:1000]}

Respond with ONLY a number 0-100."""

        resp = self.provider.generate(relevance_prompt, config.llm_model)
        relevance = extract_score(resp.text)

        # Evaluate faithfulness
        faithfulness_prompt = f"""Rate if this answer is grounded in the context (0-100):

Context: {context[:1000]}
Answer: {generated}

Respond with ONLY a number 0-100."""

        resp = self.provider.generate(faithfulness_prompt, config.llm_model)
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
            context = "\n\n".join([f"[{c.doc_id}]: {c.content}" for c in chunks])

            # Generate
            answer = self._generate(qa.question, context, config)

            # Evaluate
            scores = self._evaluate_response(qa.question, answer, qa.ground_truth, context, config)
            all_scores.append(scores)

        # Aggregate scores
        avg_correctness = np.mean([s.answer_correctness for s in all_scores])
        avg_relevance = np.mean([s.context_relevance for s in all_scores])
        avg_faithfulness = np.mean([s.faithfulness for s in all_scores])
        combined = np.mean([s.combined_score for s in all_scores])

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
