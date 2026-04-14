"""Flag generation based on evaluation results."""

from app.models.responses import (
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)


def generate_flags(
    safety: SafetyMetricResult | None,
    traceability: TraceabilityMetricResult | None,
    hallucination: MetricResult | None,
    fih_detected: list[FIHItem] | None,
    clinical_accuracy: MetricResult | None,
) -> list[str]:
    """Generate warning flags based on evaluated metrics. Skips checks for None metrics."""
    flags: list[str] = []

    if safety is not None:
        if safety.score <= 2:
            flags.append("missing_safety_data")
        if safety.missing_items:
            flags.append("safety_gaps_identified")

    if traceability is not None:
        if traceability.score <= 2:
            flags.append("poor_evidence_traceability")
        if traceability.untraced_claims:
            flags.append("untraced_claims_present")

    if hallucination is not None and hallucination.score <= 2:
        flags.append("hallucinations_detected")

    if fih_detected is not None:
        critical_fihs = [f for f in fih_detected if f.severity == "critical"]
        if critical_fihs:
            flags.append("critical_fih_detected")
        if fih_detected:
            flags.append("fih_present")

    if clinical_accuracy is not None and clinical_accuracy.score <= 2:
        flags.append("low_clinical_accuracy")

    return flags


def aggregate_confidence_level(metric_results: list) -> str:
    """Derive overall confidence from a flat list of MetricResult-like objects.

    - "low"    if any metric is low
    - "medium" if any metric is medium (and none are low)
    - "high"   if all metrics are high
    """
    levels = [r.confidence for r in metric_results if hasattr(r, "confidence")]
    if not levels:
        return "high"
    if "low" in levels:
        return "low"
    if "medium" in levels:
        return "medium"
    return "high"


def aggregate_section_scores(
    section_scores: list[dict],
    metrics: list[str],
    weight_by_length: bool = False,
) -> dict[str, float]:
    """Aggregate metric scores across sections.

    Args:
        section_scores: List of dicts, each with metric keys mapping to MetricResult-like objects.
        metrics: List of metric names to aggregate.
        weight_by_length: If True, weight scores by section content length.

    Returns:
        Dict of metric_name -> aggregated float score.
    """
    # Metrics that have numeric scores (exclude fih_detected which is a list)
    scored_metrics = [m for m in metrics if m != "fih_detected"]

    if not section_scores:
        return {}

    aggregated: dict[str, float] = {}

    for metric in scored_metrics:
        scores = []
        weights = []
        for section in section_scores:
            result = section.get(metric)
            if result is not None and hasattr(result, "score"):
                scores.append(result.score)
                weights.append(section.get("_content_length", 1))

        if not scores:
            continue

        if weight_by_length and sum(weights) > 0:
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            aggregated[metric] = round(weighted_sum / total_weight, 2)
        else:
            aggregated[metric] = round(sum(scores) / len(scores), 2)

    return aggregated
