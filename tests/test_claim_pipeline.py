"""Tests for the claim-level verification pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.metrics import METRIC_REGISTRY
from app.services.claim_pipeline import (
    _extract_claims,
    _evaluate_sub_question,
    run_percentage_metric,
)


@pytest.mark.asyncio
class TestExtractClaims:
    @patch("app.services.claim_pipeline.call_llm_judge_list")
    async def test_extract_claims_returns_valid(self, mock_llm):
        mock_llm.return_value = [
            {"claim_id": "c1", "claim_text": "Lenalidomide 25mg", "location": "Section 3"},
            {"claim_id": "c2", "claim_text": "Dexamethasone 40mg", "location": "Section 3"},
        ]
        sq = METRIC_REGISTRY["accuracy"].sub_questions[2]  # drug dosages
        claims = await _extract_claims(sq, "report text", "gpt-4o", "rpt-1")

        assert len(claims) == 2
        assert claims[0]["claim_text"] == "Lenalidomide 25mg"

    @patch("app.services.claim_pipeline.call_llm_judge_list")
    async def test_extract_claims_empty(self, mock_llm):
        mock_llm.return_value = []
        sq = METRIC_REGISTRY["accuracy"].sub_questions[0]
        claims = await _extract_claims(sq, "report text", "gpt-4o", "rpt-1")
        assert claims == []

    @patch("app.services.claim_pipeline.call_llm_judge_list")
    async def test_extract_claims_auto_assigns_id(self, mock_llm):
        """Claims without claim_id get one auto-assigned."""
        mock_llm.return_value = [
            {"claim_text": "Some claim", "location": "Section 1"},
        ]
        sq = METRIC_REGISTRY["accuracy"].sub_questions[0]
        claims = await _extract_claims(sq, "report text", "gpt-4o", "rpt-1")
        assert claims[0]["claim_id"] == "c1"


@pytest.mark.asyncio
class TestEvaluateSubQuestion:
    @patch("app.services.claim_pipeline.call_llm_judge_list")
    @patch("app.services.claim_pipeline.retrieve_for_claim")
    async def test_index_based_sub_question(self, mock_retrieve, mock_llm, sample_chunks):
        """Full flow for an index-based sub-question."""
        sq = METRIC_REGISTRY["accuracy"].sub_questions[2]  # drug dosages

        # First LLM call: extraction
        # Second LLM call: verification
        mock_llm.side_effect = [
            # Extraction result
            [{"claim_id": "c1", "claim_text": "Lenalidomide 25mg", "location": "S3"}],
            # Verification result
            [{"claim_id": "c1", "verdict": "correct", "reasoning": "Matches source"}],
        ]
        mock_retrieve.return_value = sample_chunks

        result = await _evaluate_sub_question(
            sub_question=sq,
            generated_report="Report with Lenalidomide 25mg",
            guideline_topic="NDMM treatment",
            disease_context="Multiple Myeloma",
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        assert result.sub_question_id == "accuracy_drug_dosages"
        assert result.total_count == 1
        assert result.correct_count == 1
        assert result.percentage == 100.0

    @patch("app.services.claim_pipeline.call_llm_judge_list")
    async def test_zero_claims_gives_100_percent(self, mock_llm):
        """When no claims extracted, score is 100%."""
        sq = METRIC_REGISTRY["accuracy"].sub_questions[3]  # drug interactions
        mock_llm.return_value = []

        result = await _evaluate_sub_question(
            sub_question=sq,
            generated_report="Report with no drug interactions",
            guideline_topic="topic",
            disease_context="context",
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        assert result.percentage == 100.0
        assert result.total_count == 0

    @patch("app.services.claim_pipeline.call_llm_judge_list")
    async def test_consistency_sub_question_no_retrieval(self, mock_llm):
        """Consistency sub-questions should NOT call retrieve_for_claim."""
        sq = METRIC_REGISTRY["consistency"].sub_questions[1]  # dosages across sections

        mock_llm.side_effect = [
            # Extraction
            [{"claim_id": "c1", "claim_text": "Section A says 25mg, Section B says 20mg", "location": "S1, S4"}],
            # Verification
            [{"claim_id": "c1", "verdict": "incorrect", "reasoning": "Contradiction", "conflicting_location": "Section B"}],
        ]

        with patch("app.services.claim_pipeline.retrieve_for_claim") as mock_retrieve:
            result = await _evaluate_sub_question(
                sub_question=sq,
                generated_report="Report with inconsistent dosages",
                guideline_topic="topic",
                disease_context="context",
                deployment="gpt-4o",
                report_id="rpt-1",
            )
            mock_retrieve.assert_not_called()

        assert result.correct_count == 0
        assert result.percentage == 0.0


@pytest.mark.asyncio
class TestRunPercentageMetric:
    @patch("app.services.claim_pipeline._evaluate_sub_question")
    async def test_aggregates_sub_questions(self, mock_eval):
        """Metric score is the average of sub-question percentages."""
        from app.models.responses import SubQuestionResult

        mock_eval.side_effect = [
            SubQuestionResult(
                sub_question_id="sq1", sub_question_text="Q1",
                correct_count=3, total_count=4, percentage=75.0,
            ),
            SubQuestionResult(
                sub_question_id="sq2", sub_question_text="Q2",
                correct_count=2, total_count=2, percentage=100.0,
            ),
            SubQuestionResult(
                sub_question_id="sq3", sub_question_text="Q3",
                correct_count=1, total_count=2, percentage=50.0,
            ),
            SubQuestionResult(
                sub_question_id="sq4", sub_question_text="Q4",
                correct_count=4, total_count=4, percentage=100.0,
            ),
        ]

        metric = METRIC_REGISTRY["accuracy"]
        result = await run_percentage_metric(
            metric=metric,
            generated_report="report",
            guideline_topic="topic",
            disease_context="context",
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        expected = (75.0 + 100.0 + 50.0 + 100.0) / 4
        assert result.score == round(expected, 2)
        assert len(result.sub_questions) == 4
