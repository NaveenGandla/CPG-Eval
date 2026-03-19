"""Core evaluation orchestration: majority voting and aggregation."""

import asyncio
import time
import uuid
from datetime import datetime, timezone

import structlog

from app.models.requests import EvaluationRequest
from app.models.responses import (
    EvaluationResponse,
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)
from app.prompts.evaluation_prompts import build_user_prompt, format_chunks
from app.services import blob_service, cosmos_service
from app.services.llm_judge import call_llm_judge
from app.utils.bias_mitigation import (
    aggregate_fih_detections,
    aggregate_likert_scores,
    calculate_confidence_level,
    select_median_run_index,
)
from app.utils.scoring import (
    calculate_overall_score,
    determine_usable_without_editing,
    generate_flags,
)

logger = structlog.get_logger()

LIKERT_METRICS = [
    "clinical_accuracy",
    "completeness",
    "safety_completeness",
    "relevance",
    "coherence",
    "evidence_traceability",
]
ORDINAL_METRICS = ["hallucination_score"]


async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Run the full evaluation pipeline."""
    evaluation_id = str(uuid.uuid4())
    start_time = time.monotonic()

    logger.info(
        "evaluation_start",
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        num_runs=request.num_eval_runs,
        model=request.evaluation_model,
    )

    # Build prompt
    formatted_chunks = format_chunks(request.retrieved_chunks)
    user_prompt = build_user_prompt(
        guideline_topic=request.guideline_topic,
        disease_context=request.disease_context,
        formatted_chunks=formatted_chunks,
        generated_report=request.generated_report,
    )

    # Run N independent evaluations concurrently
    tasks = [
        call_llm_judge(
            user_prompt=user_prompt,
            deployment=request.evaluation_model,
            report_id=request.report_id,
            run_index=i,
        )
        for i in range(request.num_eval_runs)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failed runs
    successful_runs: list[dict] = []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            logger.error(
                "evaluation_run_failed",
                report_id=request.report_id,
                evaluation_id=evaluation_id,
                run_index=i,
                error=str(result),
            )
        else:
            successful_runs.append(result)

    if not successful_runs:
        raise RuntimeError(
            f"All {request.num_eval_runs} evaluation runs failed for "
            f"report_id={request.report_id}"
        )

    num_successful = len(successful_runs)

    # Aggregate scores
    aggregated = _aggregate_runs(successful_runs, num_successful)

    # Calculate overall score
    overall_score = calculate_overall_score(
        clinical_accuracy=aggregated["clinical_accuracy"].score,
        completeness=aggregated["completeness"].score,
        safety_completeness=aggregated["safety_completeness"].score,
        relevance=aggregated["relevance"].score,
        coherence=aggregated["coherence"].score,
        evidence_traceability=aggregated["evidence_traceability"].score,
        hallucination_score=aggregated["hallucination_score"].score,
    )

    usable = determine_usable_without_editing(
        overall_score, aggregated["fih_detected"]
    )

    # Confidence level based on run agreement
    all_run_scores = [
        {metric: run.get(metric, {}).get("score", 0) for metric in LIKERT_METRICS + ORDINAL_METRICS}
        for run in successful_runs
    ]
    confidence = calculate_confidence_level(all_run_scores)

    flags = generate_flags(
        safety=aggregated["safety_completeness"],
        traceability=aggregated["evidence_traceability"],
        hallucination=aggregated["hallucination_score"],
        fih_detected=aggregated["fih_detected"],
        clinical_accuracy=aggregated["clinical_accuracy"],
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    # Build response
    response = EvaluationResponse(
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        timestamp=timestamp,
        model_used=request.evaluation_model,
        num_runs=num_successful,
        clinical_accuracy=aggregated["clinical_accuracy"],
        completeness=aggregated["completeness"],
        safety_completeness=aggregated["safety_completeness"],
        relevance=aggregated["relevance"],
        coherence=aggregated["coherence"],
        evidence_traceability=aggregated["evidence_traceability"],
        hallucination_score=aggregated["hallucination_score"],
        fih_detected=aggregated["fih_detected"],
        overall_score=overall_score,
        usable_without_editing=usable,
        confidence_level=confidence,
        flags=flags,
        cosmos_document_id=evaluation_id,
    )

    # Store in Cosmos DB and Blob Storage
    cosmos_doc = response.model_dump()
    cosmos_doc["id"] = evaluation_id
    cosmos_doc["_raw_runs"] = successful_runs

    try:
        await cosmos_service.store_evaluation(cosmos_doc)
    except Exception as e:
        logger.error(
            "cosmos_store_failed",
            evaluation_id=evaluation_id,
            error=str(e),
        )

    blob_url: str | None = None
    try:
        blob_data = response.model_dump()
        blob_data["_raw_runs"] = successful_runs
        blob_url = await blob_service.store_evaluation_report(
            report_id=request.report_id,
            evaluation_id=evaluation_id,
            data=blob_data,
        )
        response.blob_url = blob_url
    except Exception as e:
        logger.error(
            "blob_store_failed",
            evaluation_id=evaluation_id,
            error=str(e),
        )

    elapsed = round(time.monotonic() - start_time, 2)
    logger.info(
        "evaluation_complete",
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        overall_score=overall_score,
        confidence=confidence,
        elapsed_seconds=elapsed,
    )

    return response


def _aggregate_runs(runs: list[dict], num_runs: int) -> dict:
    """Aggregate multiple evaluation runs into final scores."""
    # Collect scores per metric
    likert_scores: dict[str, list[int]] = {m: [] for m in LIKERT_METRICS}
    ordinal_scores: dict[str, list[int]] = {m: [] for m in ORDINAL_METRICS}
    all_fihs: list[list[dict]] = []

    for run in runs:
        for metric in LIKERT_METRICS:
            entry = run.get(metric, {})
            score = entry.get("score", 3)
            likert_scores[metric].append(score)

        for metric in ORDINAL_METRICS:
            entry = run.get(metric, {})
            score = entry.get("score", 2)
            ordinal_scores[metric].append(score)

        all_fihs.append(run.get("fih_detected", []))

    result: dict = {}

    # Aggregate Likert metrics
    for metric in LIKERT_METRICS:
        scores = likert_scores[metric]
        median_idx = select_median_run_index(scores)
        median_run = runs[median_idx]
        entry = median_run.get(metric, {})
        aggregated_score = aggregate_likert_scores(scores)

        if metric == "safety_completeness":
            result[metric] = SafetyMetricResult(
                score=aggregated_score,
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
                missing_items=entry.get("missing_items", []),
            )
        elif metric == "evidence_traceability":
            result[metric] = TraceabilityMetricResult(
                score=aggregated_score,
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
                untraced_claims=entry.get("untraced_claims", []),
            )
        else:
            result[metric] = MetricResult(
                score=aggregated_score,
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
            )

    # Aggregate ordinal metrics
    for metric in ORDINAL_METRICS:
        scores = ordinal_scores[metric]
        median_idx = select_median_run_index(scores)
        median_run = runs[median_idx]
        entry = median_run.get(metric, {})
        aggregated_score = aggregate_likert_scores(scores)

        result[metric] = MetricResult(
            score=aggregated_score,
            confidence=entry.get("confidence", "medium"),
            reasoning=entry.get("reasoning", ""),
        )

    # Aggregate FIH detections via majority voting
    result["fih_detected"] = aggregate_fih_detections(all_fihs, num_runs)

    return result
