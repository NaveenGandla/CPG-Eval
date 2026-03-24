"""Tests for flag generation based on evaluation results."""

from app.models.responses import (
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)
from app.utils.scoring import generate_flags


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

    def test_flags_with_none_metrics(self):
        """When metrics are None (not selected), no flags generated for them."""
        flags = generate_flags(
            safety=None,
            traceability=None,
            hallucination=None,
            fih_detected=None,
            clinical_accuracy=MetricResult(
                score=1, confidence="low", reasoning="Bad"
            ),
        )
        assert flags == ["low_clinical_accuracy"]

    def test_all_none_metrics(self):
        flags = generate_flags(
            safety=None,
            traceability=None,
            hallucination=None,
            fih_detected=None,
            clinical_accuracy=None,
        )
        assert flags == []
