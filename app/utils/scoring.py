"""Weighted score calculation, normalization, and flag generation."""

from app.models.responses import (
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)

WEIGHTS: dict[str, float] = {
    "clinical_accuracy": 0.25,
    "completeness": 0.10,
    "safety_completeness": 0.20,
    "relevance": 0.05,
    "coherence": 0.05,
    "evidence_traceability": 0.20,
    "hallucination_score": 0.15,
}


def normalize_likert5(score: int) -> float:
    """Normalize a 1-5 Likert score to 0.0-1.0."""
    return (score - 1) / 4


def normalize_likert4(score: int) -> float:
    """Normalize a 1-4 ordinal score to 0.0-1.0."""
    return (score - 1) / 3


def calculate_overall_score(
    clinical_accuracy: int,
    completeness: int,
    safety_completeness: int,
    relevance: int,
    coherence: int,
    evidence_traceability: int,
    hallucination_score: int,
) -> float:
    """Calculate the weighted overall score (0-100)."""
    normalized = {
        "clinical_accuracy": normalize_likert5(clinical_accuracy),
        "completeness": normalize_likert5(completeness),
        "safety_completeness": normalize_likert5(safety_completeness),
        "relevance": normalize_likert5(relevance),
        "coherence": normalize_likert5(coherence),
        "evidence_traceability": normalize_likert5(evidence_traceability),
        "hallucination_score": normalize_likert4(hallucination_score),
    }

    weighted_sum = sum(normalized[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(weighted_sum * 100, 2)


def determine_usable_without_editing(
    overall_score: float, fih_detected: list[FIHItem]
) -> bool:
    """True if overall >= 80 and no critical FIH."""
    if overall_score < 80:
        return False
    return not any(fih.severity == "critical" for fih in fih_detected)


def generate_flags(
    safety: SafetyMetricResult,
    traceability: TraceabilityMetricResult,
    hallucination: MetricResult,
    fih_detected: list[FIHItem],
    clinical_accuracy: MetricResult,
) -> list[str]:
    """Generate warning flags based on evaluation results."""
    flags: list[str] = []

    if safety.score <= 2:
        flags.append("missing_safety_data")
    if safety.missing_items:
        flags.append("safety_gaps_identified")

    if traceability.score <= 2:
        flags.append("poor_evidence_traceability")
    if traceability.untraced_claims:
        flags.append("untraced_claims_present")

    if hallucination.score <= 2:
        flags.append("hallucinations_detected")

    critical_fihs = [f for f in fih_detected if f.severity == "critical"]
    if critical_fihs:
        flags.append("critical_fih_detected")
    if fih_detected:
        flags.append("fih_present")

    if clinical_accuracy.score <= 2:
        flags.append("low_clinical_accuracy")

    return flags
