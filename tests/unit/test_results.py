#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.core.experiment.results module.
"""

import pytest

from ragit.core.experiment.results import EvaluationResult, ExperimentResults


@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result."""
    return EvaluationResult(
        pattern_name="Pattern_1",
        indexing_params={"chunk_size": 256, "chunk_overlap": 50, "embedding_model": "test"},
        inference_params={"num_chunks": 3, "llm_model": "test"},
        scores={"answer_correctness": {"mean": 0.8}, "context_relevance": {"mean": 0.7}, "faithfulness": {"mean": 0.9}},
        execution_time=10.5,
        final_score=0.85,
    )


@pytest.fixture
def multiple_results():
    """Create multiple evaluation results with different scores."""
    return [
        EvaluationResult(
            pattern_name="Pattern_1",
            indexing_params={"chunk_size": 256},
            inference_params={"num_chunks": 3},
            scores={},
            execution_time=10.0,
            final_score=0.70,
        ),
        EvaluationResult(
            pattern_name="Pattern_2",
            indexing_params={"chunk_size": 512},
            inference_params={"num_chunks": 5},
            scores={},
            execution_time=15.0,
            final_score=0.90,
        ),
        EvaluationResult(
            pattern_name="Pattern_3",
            indexing_params={"chunk_size": 128},
            inference_params={"num_chunks": 2},
            scores={},
            execution_time=8.0,
            final_score=0.60,
        ),
    ]


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self, sample_evaluation_result):
        """Test creating an EvaluationResult."""
        result = sample_evaluation_result

        assert result.pattern_name == "Pattern_1"
        assert result.indexing_params["chunk_size"] == 256
        assert result.inference_params["num_chunks"] == 3
        assert result.final_score == 0.85
        assert result.execution_time == 10.5

    def test_to_dict(self, sample_evaluation_result):
        """Test converting to dictionary."""
        result = sample_evaluation_result
        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["pattern_name"] == "Pattern_1"
        assert data["final_score"] == 0.85
        assert data["indexing_params"]["chunk_size"] == 256

    def test_repr(self, sample_evaluation_result):
        """Test string representation."""
        result = sample_evaluation_result
        repr_str = repr(result)

        assert "Pattern_1" in repr_str
        assert "0.85" in repr_str or "0.850" in repr_str
        assert "10.5" in repr_str

    def test_equality(self):
        """Test EvaluationResult equality."""
        r1 = EvaluationResult(
            pattern_name="P1", indexing_params={}, inference_params={}, scores={}, execution_time=1.0, final_score=0.5
        )
        r2 = EvaluationResult(
            pattern_name="P1", indexing_params={}, inference_params={}, scores={}, execution_time=1.0, final_score=0.5
        )

        assert r1 == r2


class TestExperimentResults:
    """Tests for ExperimentResults collection class."""

    def test_empty_results(self):
        """Test empty ExperimentResults."""
        results = ExperimentResults()

        assert len(results) == 0
        assert bool(results) is False
        assert list(results) == []

    def test_add_result(self, sample_evaluation_result):
        """Test adding results."""
        results = ExperimentResults()
        results.add(sample_evaluation_result)

        assert len(results) == 1
        assert bool(results) is True

    def test_iteration(self, multiple_results):
        """Test iterating over results."""
        results = ExperimentResults(evaluations=multiple_results)

        names = [r.pattern_name for r in results]

        assert names == ["Pattern_1", "Pattern_2", "Pattern_3"]

    def test_scores_property(self, multiple_results):
        """Test scores property returns all final scores."""
        results = ExperimentResults(evaluations=multiple_results)

        scores = results.scores

        assert scores == [0.70, 0.90, 0.60]

    def test_sorted_descending(self, multiple_results):
        """Test sorting results by score (best first)."""
        results = ExperimentResults(evaluations=multiple_results)

        sorted_results = results.sorted(reverse=True)

        assert sorted_results[0].final_score == 0.90
        assert sorted_results[1].final_score == 0.70
        assert sorted_results[2].final_score == 0.60

    def test_sorted_ascending(self, multiple_results):
        """Test sorting results by score (worst first)."""
        results = ExperimentResults(evaluations=multiple_results)

        sorted_results = results.sorted(reverse=False)

        assert sorted_results[0].final_score == 0.60
        assert sorted_results[1].final_score == 0.70
        assert sorted_results[2].final_score == 0.90

    def test_get_best(self, multiple_results):
        """Test getting best k results."""
        results = ExperimentResults(evaluations=multiple_results)

        best_1 = results.get_best(k=1)
        best_2 = results.get_best(k=2)

        assert len(best_1) == 1
        assert best_1[0].pattern_name == "Pattern_2"

        assert len(best_2) == 2
        assert best_2[0].final_score == 0.90
        assert best_2[1].final_score == 0.70

    def test_is_cached_found(self, multiple_results):
        """Test is_cached returns score when found."""
        results = ExperimentResults(evaluations=multiple_results)

        score = results.is_cached(indexing_params={"chunk_size": 512}, inference_params={"num_chunks": 5})

        assert score == 0.90

    def test_is_cached_not_found(self, multiple_results):
        """Test is_cached returns None when not found."""
        results = ExperimentResults(evaluations=multiple_results)

        score = results.is_cached(indexing_params={"chunk_size": 999}, inference_params={"num_chunks": 99})

        assert score is None

    def test_len(self, multiple_results):
        """Test len() on ExperimentResults."""
        results = ExperimentResults(evaluations=multiple_results)

        assert len(results) == 3
