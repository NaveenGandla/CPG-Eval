"""Tests for evaluation engine orchestration."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.requests import ALL_METRICS, EvaluationRequest
from app.models.responses import LikertMetricResult, PercentageMetricResult
from app.services.evaluation_engine import run_evaluation


@pytest.mark.asyncio
class TestRunEvaluation:
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.run_likert_metric")
    @patch("app.services.evaluation_engine.run_percentage_metric")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_successful_evaluation(
        self,
        mock_enrich,
        mock_pct,
        mock_likert,
        mock_cosmos,
        sample_request,
        sample_chunks,
        sample_percentage_result,
        sample_likert_result,
    ):
        mock_enrich.return_value = sample_chunks
        mock_pct.return_value = sample_percentage_result
        mock_likert.return_value = sample_likert_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")

        result = await run_evaluation(sample_request)

        assert result.report_id == "rpt-test-001"
        assert result.evaluation_model == "gpt-4o"
        assert result.metrics_evaluated == ALL_METRICS
        # Percentage metrics should be set
        assert result.accuracy is not None
        assert result.accuracy.score == 75.0
        # Likert metrics should be set
        assert result.coherence is not None
        assert result.coherence.score == 3.25

    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.run_likert_metric")
    @patch("app.services.evaluation_engine.run_percentage_metric")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_selective_metrics(
        self,
        mock_enrich,
        mock_pct,
        mock_likert,
        mock_cosmos,
        sample_chunks,
        sample_percentage_result,
    ):
        """Only selected metrics are evaluated; non-selected are None."""
        request = EvaluationRequest(
            report_id="rpt-test-003",
            generated_report="Some report content here.",
            guideline_topic="Treatment for NDMM",
            disease_context="Multiple Myeloma",
            metrics=["accuracy", "consistency"],
        )
        mock_enrich.return_value = sample_chunks
        mock_pct.return_value = sample_percentage_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")

        result = await run_evaluation(request)

        assert result.metrics_evaluated == ["accuracy", "consistency"]
        assert result.accuracy is not None
        assert result.consistency is not None
        # Non-selected metrics should be None
        assert result.hallucinations is None
        assert result.coherence is None
        assert result.bias is None
        # No Likert metrics requested, so enrich_chunks should not be called
        mock_enrich.assert_not_called()

    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.run_likert_metric")
    @patch("app.services.evaluation_engine.run_percentage_metric")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_likert_only(
        self,
        mock_enrich,
        mock_pct,
        mock_likert,
        mock_cosmos,
        sample_chunks,
        sample_likert_result,
    ):
        """When only Likert metrics selected, no percentage pipeline runs."""
        request = EvaluationRequest(
            report_id="rpt-test-004",
            generated_report="Some report.",
            guideline_topic="Treatment",
            disease_context="MM",
            metrics=["coherence", "transparency"],
        )
        mock_enrich.return_value = sample_chunks
        mock_likert.return_value = sample_likert_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")

        result = await run_evaluation(request)

        assert result.coherence is not None
        assert result.transparency is not None
        assert result.accuracy is None
        mock_pct.assert_not_called()
        mock_enrich.assert_called_once()

    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.run_likert_metric")
    @patch("app.services.evaluation_engine.run_percentage_metric")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_metric_failure_handled_gracefully(
        self,
        mock_enrich,
        mock_pct,
        mock_likert,
        mock_cosmos,
        sample_request,
        sample_chunks,
        sample_likert_result,
    ):
        """If a metric pipeline raises, the evaluation still completes."""
        mock_enrich.return_value = sample_chunks
        mock_pct.side_effect = RuntimeError("LLM error")
        mock_likert.return_value = sample_likert_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")

        result = await run_evaluation(sample_request)

        # Failed percentage metrics should be None
        assert result.accuracy is None
        # Likert metrics should still succeed
        assert result.coherence is not None
