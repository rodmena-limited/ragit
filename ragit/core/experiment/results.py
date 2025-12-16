#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Ragit experiment results.
"""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """
    Result from evaluating a single RAG configuration.

    Parameters
    ----------
    pattern_name : str
        Name of the RAG pattern (e.g., "Pattern_1").
    indexing_params : dict[str, Any]
        Hyperparameters used during indexing (chunk_size, overlap, etc.).
    inference_params : dict[str, Any]
        Hyperparameters used during inference (num_chunks, llm_model, etc.).
    scores : dict[str, dict]
        Evaluation scores (answer_correctness, context_relevance, faithfulness).
    execution_time : float
        Time taken for evaluation in seconds.
    final_score : float
        Combined score for optimization ranking.
    """

    pattern_name: str
    indexing_params: dict[str, Any]
    inference_params: dict[str, Any]
    scores: dict[str, dict[str, float]]
    execution_time: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(name={self.pattern_name}, score={self.final_score:.3f}, time={self.execution_time:.1f}s)"
        )


@dataclass
class ExperimentResults:
    """
    Collection of evaluation results from an optimization experiment.

    Attributes
    ----------
    evaluations : list[EvaluationResult]
        All evaluation results.
    """

    evaluations: list[EvaluationResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.evaluations)

    def __iter__(self) -> Iterator[EvaluationResult]:
        yield from self.evaluations

    def __bool__(self) -> bool:
        return bool(self.evaluations)

    def add(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.evaluations.append(result)

    def is_cached(
        self,
        indexing_params: dict[str, Any],
        inference_params: dict[str, Any],
    ) -> float | None:
        """
        Check if this configuration was already evaluated.

        Returns
        -------
        float or None
            Final score if cached, None otherwise.
        """
        for ev in self.evaluations:
            if ev.indexing_params == indexing_params and ev.inference_params == inference_params:
                return ev.final_score
        return None

    @property
    def scores(self) -> list[float]:
        """All final scores."""
        return [ev.final_score for ev in self.evaluations]

    def sorted(self, reverse: bool = True) -> list[EvaluationResult]:
        """
        Get results sorted by final score.

        Parameters
        ----------
        reverse : bool
            If True (default), best scores first.

        Returns
        -------
        list[EvaluationResult]
            Sorted results.
        """
        return sorted(self.evaluations, key=lambda x: x.final_score, reverse=reverse)

    def get_best(self, k: int = 1) -> list[EvaluationResult]:
        """
        Get k best results.

        Parameters
        ----------
        k : int
            Number of results to return.

        Returns
        -------
        list[EvaluationResult]
            Top k results by score.
        """
        return self.sorted()[:k]
