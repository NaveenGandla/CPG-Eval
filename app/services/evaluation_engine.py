"""Core evaluation orchestration — runs percentage and Likert pipelines in parallel."""

import asyncio
import time
import uuid
from datetime import datetime, timezone

import structlog

from app.models.metrics import METRIC_REGISTRY, MetricType
from app.models.requests import EvaluationRequest
from app.models.responses import (
    EvaluationResponse,
    LikertMetricResult,
    PercentageMetricResult,
)
from app.services import cosmos_service
from app.services.claim_pipeline import run_percentage_metric
from app.services.likert_pipeline import run_likert_metric
from app.services.search_service import enrich_chunks
from app.utils.scoring import generate_flags

logger = structlog.get_logger()


async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Run the full evaluation pipeline with parallel metric evaluation."""
    evaluation_id = str(uuid.uuid4())
    start_time = time.monotonic()
    selected_metrics = list(request.metrics)

    logger.info(
        "evaluation_start",
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        evaluation_model=request.evaluation_model,
        metrics=selected_metrics,
    )

    # Categorize requested metrics
    percentage_metrics = [
        m for m in selected_metrics
        if METRIC_REGISTRY[m].metric_type == MetricType.PERCENTAGE
    ]
    likert_metrics = [
        m for m in selected_metrics
        if METRIC_REGISTRY[m].metric_type == MetricType.LIKERT
    ]

    # Retrieve document-level evidence for Likert metrics (shared across all)
    shared_chunks = []
    if likert_metrics:
        shared_chunks = await enrich_chunks(
            report_id=request.report_id,
            guideline_topic=request.guideline_topic,
            disease_context=request.disease_context,
        )

    # Build all metric evaluation tasks
    tasks: dict[str, asyncio.Task] = {}

    for metric_name in percentage_metrics:
        metric_def = METRIC_REGISTRY[metric_name]
        tasks[metric_name] = asyncio.ensure_future(
            run_percentage_metric(
                metric=metric_def,
                generated_report=request.generated_report,
                guideline_topic=request.guideline_topic,
                disease_context=request.disease_context,
                deployment=request.evaluation_model,
                report_id=request.report_id,
            )
        )

    for metric_name in likert_metrics:
        metric_def = METRIC_REGISTRY[metric_name]
        tasks[metric_name] = asyncio.ensure_future(
            run_likert_metric(
                metric=metric_def,
                generated_report=request.generated_report,
                evidence_chunks=shared_chunks,
                deployment=request.evaluation_model,
                report_id=request.report_id,
                guideline_topic=request.guideline_topic,
                disease_context=request.disease_context,
            )
        )

    # Run all metrics in parallel
    results: dict[str, PercentageMetricResult | LikertMetricResult] = {}
    if tasks:
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for metric_name, result in zip(tasks.keys(), done):
            if isinstance(result, Exception):
                logger.error(
                    "metric_evaluation_failed",
                    metric=metric_name,
                    error=str(result),
                )
            else:
                results[metric_name] = result

    # Generate flags
    flags = generate_flags(
        accuracy=results.get("accuracy"),
        hallucinations=results.get("hallucinations"),
        consistency=results.get("consistency"),
        source_traceability=results.get("source_traceability"),
        coherence=results.get("coherence"),
        clinical_relevance=results.get("clinical_relevance"),
        bias=results.get("bias"),
        transparency=results.get("transparency"),
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    response = EvaluationResponse(
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        timestamp=timestamp,
        evaluation_model=request.evaluation_model,
        metrics_evaluated=selected_metrics,
        accuracy=results.get("accuracy"),
        hallucinations=results.get("hallucinations"),
        consistency=results.get("consistency"),
        source_traceability=results.get("source_traceability"),
        coherence=results.get("coherence"),
        clinical_relevance=results.get("clinical_relevance"),
        bias=results.get("bias"),
        transparency=results.get("transparency"),
        flags=flags,
        cosmos_document_id=evaluation_id,
    )

    # Store in Cosmos DB
    cosmos_doc = response.model_dump()
    cosmos_doc["id"] = evaluation_id

    try:
        await cosmos_service.store_evaluation(cosmos_doc)
    except Exception as e:
        logger.error(
            "cosmos_store_failed",
            evaluation_id=evaluation_id,
            error=str(e),
        )

    elapsed = round(time.monotonic() - start_time, 2)
    logger.info(
        "evaluation_complete",
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        elapsed_seconds=elapsed,
        metrics_completed=list(results.keys()),
    )

    return response
