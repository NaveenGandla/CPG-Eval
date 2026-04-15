"""Tests for the Likert evaluation pipeline."""

from unittest.mock import patch

import pytest

from app.models.metrics import METRIC_REGISTRY
from app.services.likert_pipeline import run_likert_metric


@pytest.mark.asyncio
class TestRunLikertMetric:
    @patch("app.services.likert_pipeline.call_llm_judge")
    async def test_successful_evaluation(self, mock_judge, sample_chunks):
        mock_judge.return_value = {
            "sub_question_scores": [
                {"sub_question_id": "coherence_pathway_alignment", "score": 3, "reasoning": "OK"},
                {"sub_question_id": "coherence_sections", "score": 4, "reasoning": "Good"},
                {"sub_question_id": "coherence_terminology", "score": 3, "reasoning": "Fine"},
                {"sub_question_id": "coherence_unified", "score": 4, "reasoning": "Great"},
            ],
            "overall_reasoning": "Well structured.",
        }

        metric = METRIC_REGISTRY["coherence"]
        result = await run_likert_metric(
            metric=metric,
            generated_report="Some report",
            evidence_chunks=sample_chunks,
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        assert result.score == 3.5
        assert len(result.sub_questions) == 4
        assert result.overall_reasoning == "Well structured."

    @patch("app.services.likert_pipeline.call_llm_judge")
    async def test_missing_sub_question_defaults_to_2(self, mock_judge, sample_chunks):
        """If LLM omits a sub-question, it defaults to score 2."""
        mock_judge.return_value = {
            "sub_question_scores": [
                {"sub_question_id": "coherence_pathway_alignment", "score": 4, "reasoning": "Good"},
                # Missing: coherence_sections, coherence_terminology, coherence_unified
            ],
            "overall_reasoning": "Partial evaluation.",
        }

        metric = METRIC_REGISTRY["coherence"]
        result = await run_likert_metric(
            metric=metric,
            generated_report="Some report",
            evidence_chunks=sample_chunks,
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        assert len(result.sub_questions) == 4
        # 1 scored at 4, 3 defaulted to 2: (4 + 2 + 2 + 2) / 4 = 2.5
        assert result.score == 2.5

    @patch("app.services.likert_pipeline.call_llm_judge")
    async def test_score_clamped_to_range(self, mock_judge, sample_chunks):
        """Scores outside 1-4 are clamped."""
        mock_judge.return_value = {
            "sub_question_scores": [
                {"sub_question_id": "coherence_pathway_alignment", "score": 0, "reasoning": "Bad"},
                {"sub_question_id": "coherence_sections", "score": 5, "reasoning": "Great"},
                {"sub_question_id": "coherence_terminology", "score": 3, "reasoning": "OK"},
                {"sub_question_id": "coherence_unified", "score": 3, "reasoning": "OK"},
            ],
            "overall_reasoning": "Mixed.",
        }

        metric = METRIC_REGISTRY["coherence"]
        result = await run_likert_metric(
            metric=metric,
            generated_report="Some report",
            evidence_chunks=sample_chunks,
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        scores = [sq.score for sq in result.sub_questions]
        assert all(1 <= s <= 4 for s in scores)
        # 0 clamped to 1, 5 clamped to 4: (1 + 4 + 3 + 3) / 4 = 2.75
        assert result.score == 2.75

    @patch("app.services.likert_pipeline.call_llm_judge")
    async def test_empty_chunks(self, mock_judge):
        """Works with empty evidence chunks."""
        mock_judge.return_value = {
            "sub_question_scores": [
                {"sub_question_id": "bias_demographic", "score": 3, "reasoning": "Fair"},
                {"sub_question_id": "bias_guideline_priority", "score": 3, "reasoning": "OK"},
                {"sub_question_id": "bias_non_pharma", "score": 2, "reasoning": "Under-represented"},
                {"sub_question_id": "bias_conflicting_info", "score": 3, "reasoning": "OK"},
            ],
            "overall_reasoning": "Mostly unbiased.",
        }

        metric = METRIC_REGISTRY["bias"]
        result = await run_likert_metric(
            metric=metric,
            generated_report="Some report",
            evidence_chunks=[],
            deployment="gpt-4o",
            report_id="rpt-1",
        )

        assert result.score == 2.75
