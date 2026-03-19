"""Tests for evaluation engine: majority voting, score aggregation, confidence."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.requests import EvaluationRequest
from app.services.evaluation_engine import _aggregate_runs, run_evaluation
from app.utils.bias_mitigation import (
    aggregate_fih_detections,
    aggregate_likert_scores,
    calculate_confidence_level,
    select_median_run_index,
)


class TestAggregateLikertScores:
    def test_median_odd(self):
        assert aggregate_likert_scores([3, 4, 5]) == 4

    def test_median_even(self):
        # median of [3, 4] = 3.5, rounded to 4
        assert aggregate_likert_scores([3, 4]) == 4

    def test_identical_scores(self):
        assert aggregate_likert_scores([5, 5, 5]) == 5

    def test_single_score(self):
        assert aggregate_likert_scores([3]) == 3


class TestSelectMedianRunIndex:
    def test_selects_middle(self):
        assert select_median_run_index([1, 3, 5]) == 1

    def test_selects_closest(self):
        # median=3, index 1 has score 3
        assert select_median_run_index([1, 3, 4]) == 1

    def test_single(self):
        assert select_median_run_index([4]) == 0


class TestAggregateFIHDetections:
    def test_majority_vote(self):
        all_fihs = [
            [{"claim": "A", "source_says": "B", "severity": "major", "location": "p1"}],
            [{"claim": "A", "source_says": "B", "severity": "major", "location": "p1"}],
            [],
        ]
        result = aggregate_fih_detections(all_fihs, 3)
        assert len(result) == 1
        assert result[0].claim == "A"

    def test_below_threshold(self):
        all_fihs = [
            [{"claim": "A", "source_says": "B", "severity": "major", "location": "p1"}],
            [],
            [],
        ]
        result = aggregate_fih_detections(all_fihs, 3)
        assert len(result) == 0

    def test_highest_severity_selected(self):
        all_fihs = [
            [{"claim": "A", "source_says": "B", "severity": "minor", "location": "p1"}],
            [{"claim": "A", "source_says": "B", "severity": "critical", "location": "p1"}],
            [{"claim": "A", "source_says": "B", "severity": "major", "location": "p1"}],
        ]
        result = aggregate_fih_detections(all_fihs, 3)
        assert len(result) == 1
        assert result[0].severity == "critical"

    def test_empty_runs(self):
        result = aggregate_fih_detections([[], [], []], 3)
        assert result == []


class TestConfidenceLevel:
    def test_high_confidence(self):
        runs = [
            {"clinical_accuracy": 4, "completeness": 4, "safety_completeness": 3,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 3},
            {"clinical_accuracy": 4, "completeness": 4, "safety_completeness": 3,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 3},
            {"clinical_accuracy": 4, "completeness": 5, "safety_completeness": 3,
             "relevance": 5, "coherence": 4, "evidence_traceability": 4,
             "hallucination_score": 3},
        ]
        assert calculate_confidence_level(runs) == "high"

    def test_medium_confidence(self):
        runs = [
            {"clinical_accuracy": 3, "completeness": 3, "safety_completeness": 3,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 3},
            {"clinical_accuracy": 5, "completeness": 3, "safety_completeness": 3,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 3},
        ]
        assert calculate_confidence_level(runs) == "medium"

    def test_low_confidence(self):
        runs = [
            {"clinical_accuracy": 1, "completeness": 5, "safety_completeness": 1,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 1},
            {"clinical_accuracy": 5, "completeness": 5, "safety_completeness": 5,
             "relevance": 5, "coherence": 4, "evidence_traceability": 3,
             "hallucination_score": 4},
        ]
        assert calculate_confidence_level(runs) == "low"


class TestAggregateRuns:
    def test_aggregate_three_runs(self, sample_llm_judge_result):
        # Create 3 slightly different runs
        run1 = sample_llm_judge_result.copy()
        run2 = sample_llm_judge_result.copy()
        run2["clinical_accuracy"] = {"score": 5, "confidence": "high", "reasoning": "Excellent."}
        run3 = sample_llm_judge_result.copy()
        run3["clinical_accuracy"] = {"score": 3, "confidence": "medium", "reasoning": "Average."}

        result = _aggregate_runs([run1, run2, run3], 3)

        # Median of [4, 5, 3] = 4
        assert result["clinical_accuracy"].score == 4
        assert result["completeness"].score == 3
        assert result["hallucination_score"].score == 3
        assert isinstance(result["fih_detected"], list)


@pytest.mark.asyncio
class TestRunEvaluation:
    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    async def test_successful_evaluation(
        self,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
        sample_llm_judge_result: dict,
    ):
        mock_judge.return_value = sample_llm_judge_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(
            return_value="https://blob.example.com/report.json"
        )

        result = await run_evaluation(sample_request)

        assert result.report_id == "rpt-test-001"
        assert result.num_runs == 3
        assert 0 <= result.overall_score <= 100
        assert result.confidence_level in ("high", "medium", "low")
        assert mock_judge.call_count == 3

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    async def test_partial_failure(
        self,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
        sample_llm_judge_result: dict,
    ):
        """If some runs fail, evaluation still succeeds with remaining runs."""
        mock_judge.side_effect = [
            sample_llm_judge_result,
            ValueError("LLM error"),
            sample_llm_judge_result,
        ]
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_evaluation(sample_request)

        assert result.num_runs == 2  # Only 2 successful

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    async def test_all_runs_fail(
        self,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
    ):
        mock_judge.side_effect = ValueError("LLM error")
        mock_cosmos.store_evaluation = AsyncMock()
        mock_blob.store_evaluation_report = AsyncMock()

        with pytest.raises(RuntimeError, match="All .* evaluation runs failed"):
            await run_evaluation(sample_request)
