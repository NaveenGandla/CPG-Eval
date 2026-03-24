"""Tests for evaluation engine: metric extraction, selective metrics, and single-run evaluation."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.requests import ALL_METRICS, EvaluationRequest, SourceChunk, SourceChunkMetadata
from app.services.evaluation_engine import _extract_metrics, run_evaluation


class TestExtractMetrics:
    def test_extract_all_metrics(self, sample_llm_judge_result):
        result = _extract_metrics(sample_llm_judge_result, ALL_METRICS)

        assert result["clinical_accuracy"].score == 4
        assert result["completeness"].score == 3
        assert result["safety_completeness"].score == 2
        assert result["safety_completeness"].missing_items == [
            "Grade 3-4 AE rates",
            "Black box warnings",
        ]
        assert result["relevance"].score == 5
        assert result["coherence"].score == 4
        assert result["evidence_traceability"].score == 3
        assert len(result["evidence_traceability"].untraced_claims) == 1
        assert result["hallucination_score"].score == 3
        assert isinstance(result["fih_detected"], list)

    def test_extract_subset_metrics(self, sample_llm_judge_result):
        """Only extract selected metrics."""
        selected = ["clinical_accuracy", "hallucination_score"]
        result = _extract_metrics(sample_llm_judge_result, selected)

        assert "clinical_accuracy" in result
        assert result["clinical_accuracy"].score == 4
        assert "hallucination_score" in result
        assert result["hallucination_score"].score == 3
        # Non-selected metrics should be absent
        assert "completeness" not in result
        assert "safety_completeness" not in result
        assert "fih_detected" not in result

    def test_extract_only_fih(self, sample_llm_judge_result):
        result = _extract_metrics(sample_llm_judge_result, ["fih_detected"])
        assert "fih_detected" in result
        assert isinstance(result["fih_detected"], list)
        assert "clinical_accuracy" not in result

    def test_extract_with_fih(self, sample_llm_judge_result):
        run = sample_llm_judge_result.copy()
        run["fih_detected"] = [
            {
                "claim": "ORR was 100%",
                "source_says": "ORR was 99%",
                "severity": "major",
                "location": "paragraph 1",
            }
        ]
        result = _extract_metrics(run, ALL_METRICS)
        assert len(result["fih_detected"]) == 1
        assert result["fih_detected"][0].claim == "ORR was 100%"
        assert result["fih_detected"][0].severity == "major"

    def test_extract_defaults_on_missing(self):
        """Missing metrics should use defaults."""
        result = _extract_metrics({}, ALL_METRICS)
        assert result["clinical_accuracy"].score == 3
        assert result["hallucination_score"].score == 2
        assert result["fih_detected"] == []


@pytest.mark.asyncio
class TestRunEvaluation:
    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_successful_evaluation(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
        sample_llm_judge_result: dict,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.return_value = sample_llm_judge_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(
            return_value="https://blob.example.com/report.json"
        )

        result = await run_evaluation(sample_request)

        assert result.report_id == "rpt-test-001"
        assert result.evaluation_model == "gpt-4o"
        assert result.num_runs == 1
        assert result.metrics_evaluated == ALL_METRICS
        assert result.confidence_level == "high"
        assert mock_judge.call_count == 1
        mock_search.assert_called_once()

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_selective_metrics(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_chunks,
    ):
        """Only selected metrics are evaluated; non-selected are None."""
        request = EvaluationRequest(
            report_id="rpt-test-003",
            generated_report="Some report content here.",
            guideline_topic="Treatment for NDMM",
            disease_context="Multiple Myeloma",
            model="gpt-4o",
            metrics=["clinical_accuracy", "safety_completeness", "fih_detected"],
        )
        llm_result = {
            "clinical_accuracy": {"score": 4, "confidence": "high", "reasoning": "Good."},
            "safety_completeness": {
                "score": 3,
                "confidence": "medium",
                "reasoning": "OK.",
                "missing_items": [],
            },
            "fih_detected": [],
        }
        mock_search.return_value = sample_chunks
        mock_judge.return_value = llm_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_evaluation(request)

        assert result.metrics_evaluated == [
            "clinical_accuracy",
            "safety_completeness",
            "fih_detected",
        ]
        assert result.clinical_accuracy is not None
        assert result.clinical_accuracy.score == 4
        assert result.safety_completeness is not None
        assert result.fih_detected is not None
        # Non-selected metrics should be None
        assert result.completeness is None
        assert result.relevance is None
        assert result.coherence is None
        assert result.evidence_traceability is None
        assert result.hallucination_score is None

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_fih_only_no_overall_score(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_chunks,
    ):
        """When only fih_detected is selected, overall_score is None."""
        request = EvaluationRequest(
            report_id="rpt-test-004",
            generated_report="Some report.",
            guideline_topic="Treatment",
            disease_context="MM",
            model="gpt-4o",
            metrics=["fih_detected"],
        )
        mock_search.return_value = sample_chunks
        mock_judge.return_value = {"fih_detected": []}
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_evaluation(request)

        assert result.fih_detected == []
        assert result.clinical_accuracy is None

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_run_fails(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.side_effect = ValueError("LLM error")
        mock_cosmos.store_evaluation = AsyncMock()
        mock_blob.store_evaluation_report = AsyncMock()

        with pytest.raises(RuntimeError, match="Evaluation run failed"):
            await run_evaluation(sample_request)

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.enrich_chunks")
    async def test_no_chunks_retrieved(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_request: EvaluationRequest,
    ):
        mock_search.return_value = []
        mock_cosmos.store_evaluation = AsyncMock()
        mock_blob.store_evaluation_report = AsyncMock()

        with pytest.raises(RuntimeError, match="No source chunks retrieved"):
            await run_evaluation(sample_request)

        mock_judge.assert_not_called()

