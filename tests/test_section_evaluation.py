"""Tests for section-wise evaluation pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.requests import (
    ALL_METRICS,
    ReportJSON,
    ReportSection,
    SectionEvaluationRequest,
    SourceChunk,
    SourceChunkMetadata,
)
from app.services.evaluation_engine import run_section_evaluation
from app.utils.scoring import aggregate_section_scores
from app.models.responses import MetricResult, SafetyMetricResult


@pytest.fixture
def sample_report_json() -> ReportJSON:
    return ReportJSON(
        report_id="rpt-section-001",
        sections=[
            ReportSection(
                id="sec-1",
                title="Introduction",
                content="This guideline covers first-line treatment for NDMM.",
                section_type="general",
                order=0,
                keywords=["NDMM", "treatment", "first-line"],
            ),
            ReportSection(
                id="sec-2",
                title="Treatment Recommendations",
                content="D-VRd is recommended as first-line therapy. The GRIFFIN trial showed 99% ORR.",
                section_type="guideline",
                order=1,
                keywords=["D-VRd", "GRIFFIN", "ORR", "therapy"],
            ),
        ],
    )


@pytest.fixture
def sample_section_request(sample_report_json) -> SectionEvaluationRequest:
    return SectionEvaluationRequest(
        guideline_topic="First-line treatment for transplant-eligible NDMM",
        disease_context="Multiple Myeloma",
        report_json=sample_report_json,
    )


@pytest.fixture
def sample_section_llm_result() -> dict:
    return {
        "clinical_accuracy": {
            "score": 4,
            "confidence": "high",
            "reasoning": "Section content verified.",
        },
        "completeness": {
            "score": 3,
            "confidence": "medium",
            "reasoning": "Covers main points.",
        },
        "safety_completeness": {
            "score": 3,
            "confidence": "medium",
            "reasoning": "Some safety info.",
            "missing_items": [],
        },
        "relevance": {
            "score": 5,
            "confidence": "high",
            "reasoning": "On-topic.",
        },
        "coherence": {
            "score": 4,
            "confidence": "high",
            "reasoning": "Well-written.",
        },
        "evidence_traceability": {
            "score": 3,
            "confidence": "medium",
            "reasoning": "Mostly traceable.",
            "untraced_claims": [],
        },
        "hallucination_score": {
            "score": 3,
            "confidence": "high",
            "reasoning": "No hallucinations.",
        },
        "fih_detected": [],
    }


@pytest.fixture
def sample_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="chunk-1",
            text="D-VRd showed 99% ORR in NDMM patients.",
            metadata=SourceChunkMetadata(study_name="GRIFFIN", year=2023),
        ),
    ]


@pytest.mark.asyncio
class TestRunSectionEvaluation:
    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_successful_section_evaluation(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_section_request,
        sample_section_llm_result,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.return_value = sample_section_llm_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_section_evaluation(sample_section_request)

        assert result.report_id == "rpt-section-001"
        assert len(result.section_scores) == 2
        assert result.metrics_evaluated == ALL_METRICS
        assert result.confidence_level == "high"
        # Should have been called once per section
        assert mock_judge.call_count == 2
        assert mock_search.call_count == 2

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_section_scores_populated(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_section_request,
        sample_section_llm_result,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.return_value = sample_section_llm_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_section_evaluation(sample_section_request)

        # Check first section score
        sec1 = result.section_scores[0]
        assert sec1.section_id == "sec-1"
        assert sec1.section_title == "Introduction"
        assert sec1.clinical_accuracy is not None
        assert sec1.clinical_accuracy.score == 4

        # Check second section score
        sec2 = result.section_scores[1]
        assert sec2.section_id == "sec-2"
        assert sec2.section_title == "Treatment Recommendations"

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_aggregated_final_scores(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_section_request,
        sample_section_llm_result,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.return_value = sample_section_llm_result
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_section_evaluation(sample_section_request)

        # All sections return same scores, so average equals same score
        assert "clinical_accuracy" in result.final_scores
        assert result.final_scores["clinical_accuracy"] == 4.0
        assert result.final_scores["relevance"] == 5.0

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_selective_metrics(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_report_json,
        sample_chunks,
    ):
        request = SectionEvaluationRequest(
            guideline_topic="Treatment for NDMM",
            disease_context="Multiple Myeloma",
            report_json=sample_report_json,
            metrics=["clinical_accuracy", "hallucination_score"],
        )
        mock_search.return_value = sample_chunks
        mock_judge.return_value = {
            "clinical_accuracy": {"score": 4, "confidence": "high", "reasoning": "Good."},
            "hallucination_score": {"score": 3, "confidence": "high", "reasoning": "Clean."},
        }
        mock_cosmos.store_evaluation = AsyncMock(return_value="doc-id")
        mock_blob.store_evaluation_report = AsyncMock(return_value=None)

        result = await run_section_evaluation(request)

        assert result.metrics_evaluated == ["clinical_accuracy", "hallucination_score"]
        sec = result.section_scores[0]
        assert sec.clinical_accuracy is not None
        assert sec.completeness is None

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_section_evaluation_failure(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
        sample_section_request,
        sample_chunks,
    ):
        mock_search.return_value = sample_chunks
        mock_judge.side_effect = ValueError("LLM error")
        mock_cosmos.store_evaluation = AsyncMock()
        mock_blob.store_evaluation_report = AsyncMock()

        with pytest.raises(RuntimeError, match="Section evaluation failed"):
            await run_section_evaluation(sample_section_request)

    @patch("app.services.evaluation_engine.blob_service")
    @patch("app.services.evaluation_engine.cosmos_service")
    @patch("app.services.evaluation_engine.call_llm_judge")
    @patch("app.services.evaluation_engine.retrieve_for_section")
    async def test_no_input_raises_value_error(
        self,
        mock_search,
        mock_judge,
        mock_cosmos,
        mock_blob,
    ):
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
        )
        with pytest.raises(ValueError, match="No input provided"):
            await run_section_evaluation(request)


class TestAggregateScores:
    def test_average_aggregation(self):
        section_scores = [
            {
                "clinical_accuracy": MetricResult(score=4, confidence="high", reasoning="Good"),
                "completeness": MetricResult(score=3, confidence="medium", reasoning="OK"),
            },
            {
                "clinical_accuracy": MetricResult(score=2, confidence="low", reasoning="Bad"),
                "completeness": MetricResult(score=5, confidence="high", reasoning="Great"),
            },
        ]
        result = aggregate_section_scores(
            section_scores, ["clinical_accuracy", "completeness"]
        )
        assert result["clinical_accuracy"] == 3.0
        assert result["completeness"] == 4.0

    def test_weighted_aggregation(self):
        section_scores = [
            {
                "clinical_accuracy": MetricResult(score=4, confidence="high", reasoning="Good"),
                "_content_length": 100,
            },
            {
                "clinical_accuracy": MetricResult(score=2, confidence="low", reasoning="Bad"),
                "_content_length": 300,
            },
        ]
        result = aggregate_section_scores(
            section_scores, ["clinical_accuracy"], weight_by_length=True
        )
        # Weighted: (4*100 + 2*300) / 400 = 1000/400 = 2.5
        assert result["clinical_accuracy"] == 2.5

    def test_empty_sections(self):
        result = aggregate_section_scores([], ["clinical_accuracy"])
        assert result == {}

    def test_fih_excluded_from_aggregation(self):
        section_scores = [
            {
                "clinical_accuracy": MetricResult(score=4, confidence="high", reasoning="Good"),
            }
        ]
        result = aggregate_section_scores(
            section_scores, ["clinical_accuracy", "fih_detected"]
        )
        assert "fih_detected" not in result
        assert "clinical_accuracy" in result
