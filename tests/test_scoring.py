"""Tests for flag generation based on evaluation results."""

from app.models.responses import (
    LikertMetricResult,
    LikertSubQuestionScore,
    PercentageMetricResult,
    SubQuestionResult,
)
from app.utils.scoring import generate_flags


class TestGenerateFlags:
    def test_no_flags_good_scores(self):
        flags = generate_flags(
            accuracy=PercentageMetricResult(score=90.0),
            hallucinations=PercentageMetricResult(score=95.0),
            consistency=PercentageMetricResult(score=90.0),
            source_traceability=PercentageMetricResult(score=80.0),
            coherence=LikertMetricResult(score=3.5, overall_reasoning="Good"),
            clinical_relevance=LikertMetricResult(score=3.0, overall_reasoning="Good"),
            bias=LikertMetricResult(score=3.0, overall_reasoning="Good"),
            transparency=LikertMetricResult(score=3.0, overall_reasoning="Good"),
        )
        assert flags == []

    def test_flags_low_accuracy(self):
        flags = generate_flags(
            accuracy=PercentageMetricResult(score=50.0),
        )
        assert "low_accuracy" in flags

    def test_flags_high_hallucination_rate(self):
        flags = generate_flags(
            hallucinations=PercentageMetricResult(score=60.0),
        )
        assert "high_hallucination_rate" in flags

    def test_flags_inconsistencies(self):
        flags = generate_flags(
            consistency=PercentageMetricResult(score=70.0),
        )
        assert "inconsistencies_detected" in flags

    def test_flags_poor_traceability(self):
        flags = generate_flags(
            source_traceability=PercentageMetricResult(score=40.0),
        )
        assert "poor_source_traceability" in flags

    def test_flags_low_likert(self):
        flags = generate_flags(
            coherence=LikertMetricResult(score=1.5, overall_reasoning="Poor"),
            bias=LikertMetricResult(score=1.8, overall_reasoning="Biased"),
        )
        assert "low_coherence" in flags
        assert "bias_detected" in flags

    def test_flags_with_none_metrics(self):
        """When metrics are None (not selected), no flags generated for them."""
        flags = generate_flags(
            accuracy=PercentageMetricResult(score=40.0),
            hallucinations=None,
            consistency=None,
        )
        assert flags == ["low_accuracy"]

    def test_all_none_metrics(self):
        flags = generate_flags()
        assert flags == []

    def test_critical_dosage_flag(self):
        """Sub-question level flag for drug dosage accuracy below 50%."""
        accuracy = PercentageMetricResult(
            score=55.0,
            sub_questions=[
                SubQuestionResult(
                    sub_question_id="accuracy_drug_dosages",
                    sub_question_text="Are drug dosages correct?",
                    correct_count=1,
                    total_count=4,
                    percentage=25.0,
                ),
            ],
        )
        flags = generate_flags(accuracy=accuracy)
        assert "low_accuracy" in flags
        assert "critical_dosage_accuracy_issue" in flags

    def test_fake_citations_flag(self):
        """Sub-question level flag for fake citations below 50%."""
        hallucinations = PercentageMetricResult(
            score=60.0,
            sub_questions=[
                SubQuestionResult(
                    sub_question_id="hallucination_fake_citations",
                    sub_question_text="Are fake citations created?",
                    correct_count=1,
                    total_count=5,
                    percentage=20.0,
                ),
            ],
        )
        flags = generate_flags(hallucinations=hallucinations)
        assert "high_hallucination_rate" in flags
        assert "fake_citations_detected" in flags
