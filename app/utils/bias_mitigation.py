"""Majority voting logic for aggregating multiple evaluation runs."""

import math
from collections import Counter
from statistics import median

from app.models.responses import FIHItem


def aggregate_likert_scores(scores: list[int]) -> int:
    """Take the median of Likert scores across N runs, rounded to nearest int."""
    return round(median(scores))


def aggregate_confidence(confidences: list[str]) -> str:
    """Take the most common confidence level (majority vote)."""
    counter = Counter(confidences)
    return counter.most_common(1)[0][0]


def aggregate_reasoning(reasonings: list[str]) -> str:
    """Pick the reasoning from the run closest to the median score.

    Since we don't have scores here, just pick the first one as a simple heuristic.
    The caller should pass the reasoning from the median-scoring run.
    """
    return reasonings[0]


def select_median_run_index(scores: list[int]) -> int:
    """Return the index of the run whose score is closest to the median."""
    med = median(scores)
    min_diff = float("inf")
    best_idx = 0
    for i, s in enumerate(scores):
        diff = abs(s - med)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    return best_idx


def aggregate_fih_detections(
    all_fihs: list[list[dict]], num_runs: int
) -> list[FIHItem]:
    """A claim is flagged only if detected in >= ceil(N/2) runs.

    Claims are matched by their 'claim' text (exact match).
    """
    threshold = math.ceil(num_runs / 2)
    claim_counts: dict[str, list[dict]] = {}

    for run_fihs in all_fihs:
        seen_in_run: set[str] = set()
        for fih in run_fihs:
            claim_text = fih.get("claim", "")
            if claim_text in seen_in_run:
                continue
            seen_in_run.add(claim_text)
            if claim_text not in claim_counts:
                claim_counts[claim_text] = []
            claim_counts[claim_text].append(fih)

    result: list[FIHItem] = []
    for claim_text, occurrences in claim_counts.items():
        if len(occurrences) >= threshold:
            # Use the first occurrence's details but pick the highest severity
            severity_order = {"critical": 3, "major": 2, "minor": 1}
            best = max(
                occurrences, key=lambda f: severity_order.get(f.get("severity", "minor"), 0)
            )
            result.append(
                FIHItem(
                    claim=best.get("claim", ""),
                    source_says=best.get("source_says", ""),
                    severity=best.get("severity", "minor"),
                    location=best.get("location", ""),
                )
            )

    return result


def calculate_confidence_level(all_run_scores: list[dict[str, int]]) -> str:
    """Determine confidence based on agreement across runs.

    - "high": all N runs agree within +/-1 on ALL metrics
    - "medium": all N runs agree within +/-2 on ALL metrics
    - "low": otherwise
    """
    metrics = [
        "clinical_accuracy",
        "completeness",
        "safety_completeness",
        "relevance",
        "coherence",
        "evidence_traceability",
        "hallucination_score",
    ]

    max_spread = 0
    for metric in metrics:
        scores = [run.get(metric, 0) for run in all_run_scores]
        if scores:
            spread = max(scores) - min(scores)
            max_spread = max(max_spread, spread)

    if max_spread <= 1:
        return "high"
    elif max_spread <= 2:
        return "medium"
    else:
        return "low"
