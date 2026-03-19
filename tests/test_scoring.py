"""Tests for weighted score calculation, normalization, and flag generation."""

from app.models.responses import (
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)
from app.utils.scoring import (
    calculate_overall_score,
    determine_usable_without_editing,
    generate_flags,
    normalize_likert4,
    normalize_likert5,
)


class TestNormalization:
    def test_likert5_min(self):
        assert normalize_likert5(1) == 0.0

    def test_likert5_max(self):
        assert normalize_likert5(5) == 1.0

    def test_likert5_mid(self):
        assert normalize_likert5(3) == 0.5

    def test_likert4_min(self):
        assert normalize_likert4(1) == 0.0

    def test_likert4_max(self):
        assert normalize_likert4(4) == 1.0

    def test_likert4_mid(self):
        assert abs(normalize_likert4(2) - 1 / 3) < 1e-9


class TestOverallScore:
    def test_perfect_scores(self):
        score = calculate_overall_score(
            clinical_accuracy=5,
            completeness=5,
            safety_completeness=5,
            relevance=5,
            coherence=5,
            evidence_traceability=5,
            hallucination_score=4,
        )
        assert score == 100.0

    def test_minimum_scores(self):
        score = calculate_overall_score(
            clinical_accuracy=1,
            completeness=1,
            safety_completeness=1,
            relevance=1,
            coherence=1,
            evidence_traceability=1,
            hallucination_score=1,
        )
        assert score == 0.0

    def test_mixed_scores(self):
        score = calculate_overall_score(
            clinical_accuracy=4,
            completeness=3,
            safety_completeness=2,
            relevance=5,
            coherence=4,
            evidence_traceability=3,
            hallucination_score=3,
        )
        assert 0 <= score <= 100
        # Verify it's calculated correctly
        expected = round(
            (
                (3 / 4) * 0.25
                + (2 / 4) * 0.10
                + (1 / 4) * 0.20
                + (4 / 4) * 0.05
                + (3 / 4) * 0.05
                + (2 / 4) * 0.20
                + (2 / 3) * 0.15
            )
            * 100,
            2,
        )
        assert score == expected


class TestUsableWithoutEditing:
    def test_usable_high_score_no_critical_fih(self):
        assert determine_usable_without_editing(85.0, []) is True

    def test_not_usable_low_score(self):
        assert determine_usable_without_editing(75.0, []) is False

    def test_not_usable_critical_fih(self):
        fih = FIHItem(
            claim="test",
            source_says="wrong",
            severity="critical",
            location="p1",
        )
        assert determine_usable_without_editing(90.0, [fih]) is False

    def test_usable_minor_fih_high_score(self):
        fih = FIHItem(
            claim="test",
            source_says="slightly off",
            severity="minor",
            location="p1",
        )
        assert determine_usable_without_editing(85.0, [fih]) is True


class TestGenerateFlags:
    def test_no_flags_good_scores(self):
        flags = generate_flags(
            safety=SafetyMetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
            traceability=TraceabilityMetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
            hallucination=MetricResult(
                score=3, confidence="high", reasoning="Good"
            ),
            fih_detected=[],
            clinical_accuracy=MetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
        )
        assert flags == []

    def test_flags_poor_safety(self):
        flags = generate_flags(
            safety=SafetyMetricResult(
                score=1,
                confidence="low",
                reasoning="Bad",
                missing_items=["AE rates"],
            ),
            traceability=TraceabilityMetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
            hallucination=MetricResult(
                score=3, confidence="high", reasoning="Good"
            ),
            fih_detected=[],
            clinical_accuracy=MetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
        )
        assert "missing_safety_data" in flags
        assert "safety_gaps_identified" in flags

    def test_flags_critical_fih(self):
        fih = FIHItem(
            claim="test",
            source_says="wrong",
            severity="critical",
            location="p1",
        )
        flags = generate_flags(
            safety=SafetyMetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
            traceability=TraceabilityMetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
            hallucination=MetricResult(
                score=3, confidence="high", reasoning="Good"
            ),
            fih_detected=[fih],
            clinical_accuracy=MetricResult(
                score=4, confidence="high", reasoning="Good"
            ),
        )
        assert "critical_fih_detected" in flags
        assert "fih_present" in flags
