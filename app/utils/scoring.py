"""Flag generation based on evaluation results."""

from app.models.responses import LikertMetricResult, PercentageMetricResult

# Thresholds for percentage metrics (below this → flag)
PERCENTAGE_THRESHOLDS: dict[str, tuple[str, float]] = {
    "accuracy": ("low_accuracy", 60.0),
    "hallucinations": ("high_hallucination_rate", 70.0),
    "consistency": ("inconsistencies_detected", 80.0),
    "source_traceability": ("poor_source_traceability", 60.0),
}

# Thresholds for Likert metrics (below this → flag)
LIKERT_THRESHOLDS: dict[str, tuple[str, float]] = {
    "coherence": ("low_coherence", 2.0),
    "clinical_relevance": ("low_clinical_relevance", 2.0),
    "bias": ("bias_detected", 2.0),
    "transparency": ("low_transparency", 2.0),
}


def generate_flags(
    accuracy: PercentageMetricResult | None = None,
    hallucinations: PercentageMetricResult | None = None,
    consistency: PercentageMetricResult | None = None,
    source_traceability: PercentageMetricResult | None = None,
    coherence: LikertMetricResult | None = None,
    clinical_relevance: LikertMetricResult | None = None,
    bias: LikertMetricResult | None = None,
    transparency: LikertMetricResult | None = None,
) -> list[str]:
    """Generate warning flags based on metric scores and thresholds."""
    flags: list[str] = []

    percentage_results = {
        "accuracy": accuracy,
        "hallucinations": hallucinations,
        "consistency": consistency,
        "source_traceability": source_traceability,
    }

    likert_results = {
        "coherence": coherence,
        "clinical_relevance": clinical_relevance,
        "bias": bias,
        "transparency": transparency,
    }

    # Check percentage metrics
    for metric_name, result in percentage_results.items():
        if result is None:
            continue
        flag_name, threshold = PERCENTAGE_THRESHOLDS[metric_name]
        if result.score < threshold:
            flags.append(flag_name)

    # Check Likert metrics
    for metric_name, result in likert_results.items():
        if result is None:
            continue
        flag_name, threshold = LIKERT_THRESHOLDS[metric_name]
        if result.score < threshold:
            flags.append(flag_name)

    # Drill into sub-question details for critical flags
    if accuracy is not None:
        for sq in accuracy.sub_questions:
            if sq.sub_question_id == "accuracy_drug_dosages" and sq.percentage < 50.0:
                flags.append("critical_dosage_accuracy_issue")
            if sq.sub_question_id == "accuracy_drug_interactions" and sq.percentage < 50.0:
                flags.append("critical_drug_interaction_issue")

    if hallucinations is not None:
        for sq in hallucinations.sub_questions:
            if sq.sub_question_id == "hallucination_fake_citations" and sq.percentage < 50.0:
                flags.append("fake_citations_detected")

    return flags
