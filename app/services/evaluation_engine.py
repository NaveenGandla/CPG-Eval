"""Core evaluation orchestration — supports both full-document and section-wise modes."""

import time
import uuid
from datetime import datetime, timezone

import structlog

from app.config import settings
from app.models.requests import (
    EvaluationRequest,
    ReportJSON,
    SectionEvaluationRequest,
)
from app.models.responses import (
    EvaluationResponse,
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    SectionEvaluationResponse,
    SectionScore,
    TraceabilityMetricResult,
)
from app.prompts.evaluation_prompts import (
    build_section_system_prompt,
    build_section_user_prompt,
    build_system_prompt,
    build_user_prompt,
    format_chunks,
)
from app.services import blob_service, cosmos_service
from app.services.input_resolver import resolve_to_json
from app.services.llm_judge import call_llm_judge
from app.services.search_service import enrich_chunks, retrieve_for_section
from app.utils.scoring import aggregate_confidence_level, aggregate_section_scores, generate_flags

logger = structlog.get_logger()


async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """Run the full evaluation pipeline."""
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

    # Retrieve source chunks from Azure AI Search
    retrieved_chunks = await enrich_chunks(
        report_id=request.report_id,
        guideline_topic=request.guideline_topic,
        disease_context=request.disease_context,
    )

    if not retrieved_chunks:
        raise RuntimeError(
            f"No source chunks retrieved from Azure AI Search for "
            f"report_id={request.report_id}, topic='{request.guideline_topic}'"
        )

    # Build prompts — system prompt is tailored to selected metrics
    system_prompt = build_system_prompt(selected_metrics)
    formatted_chunks = format_chunks(retrieved_chunks)
    user_prompt = build_user_prompt(
        guideline_topic=request.guideline_topic,
        disease_context=request.disease_context,
        formatted_chunks=formatted_chunks,
        generated_report=request.generated_report,
    )

    # Run a single evaluation
    try:
        raw_result = await call_llm_judge(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            deployment=request.evaluation_model,
            report_id=request.report_id,
            run_index=0,
        )
    except Exception as e:
        logger.error(
            "evaluation_run_failed",
            report_id=request.report_id,
            evaluation_id=evaluation_id,
            error=str(e),
        )
        raise RuntimeError(
            f"Evaluation run failed for report_id={request.report_id}: {e}"
        )

    # Extract only the selected metrics from the LLM result
    result = _extract_metrics(raw_result, selected_metrics)

    confidence = aggregate_confidence_level(
        [v for v in result.values() if hasattr(v, "confidence")]
    )

    flags = generate_flags(
        safety=result.get("safety_completeness"),
        traceability=result.get("evidence_traceability"),
        hallucination=result.get("hallucination_score"),
        fih_detected=result.get("fih_detected"),
        clinical_accuracy=result.get("clinical_accuracy"),
    )

    timestamp = datetime.now(timezone.utc).isoformat()

    # Build response
    response = EvaluationResponse(
        report_id=request.report_id,
        evaluation_id=evaluation_id,
        timestamp=timestamp,
        evaluation_model=request.evaluation_model,
        num_runs=1,
        metrics_evaluated=selected_metrics,
        clinical_accuracy=result.get("clinical_accuracy"),
        completeness=result.get("completeness"),
        safety_completeness=result.get("safety_completeness"),
        relevance=result.get("relevance"),
        coherence=result.get("coherence"),
        evidence_traceability=result.get("evidence_traceability"),
        hallucination_score=result.get("hallucination_score"),
        fih_detected=result.get("fih_detected"),
        confidence_level=confidence,
        flags=flags,
        cosmos_document_id=evaluation_id,
    )

    # Store in Cosmos DB and Blob Storage
    cosmos_doc = response.model_dump()
    cosmos_doc["id"] = evaluation_id
    cosmos_doc["_raw_result"] = raw_result

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
        blob_data["_raw_result"] = raw_result
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
        confidence=confidence,
        elapsed_seconds=elapsed,
    )

    return response


def _extract_metrics(run: dict, selected_metrics: list[str]) -> dict:
    """Extract typed metric results from a single LLM judge run for selected metrics only."""
    result: dict = {}

    likert_metrics = [
        "clinical_accuracy",
        "completeness",
        "safety_completeness",
        "relevance",
        "coherence",
        "evidence_traceability",
    ]

    for metric in likert_metrics:
        if metric not in selected_metrics:
            continue
        entry = run.get(metric, {})
        if metric == "safety_completeness":
            result[metric] = SafetyMetricResult(
                score=entry.get("score", 3),
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
                missing_items=entry.get("missing_items", []),
            )
        elif metric == "evidence_traceability":
            result[metric] = TraceabilityMetricResult(
                score=entry.get("score", 3),
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
                untraced_claims=entry.get("untraced_claims", []),
            )
        else:
            result[metric] = MetricResult(
                score=entry.get("score", 3),
                confidence=entry.get("confidence", "medium"),
                reasoning=entry.get("reasoning", ""),
            )

    if "hallucination_score" in selected_metrics:
        hal_entry = run.get("hallucination_score", {})
        result["hallucination_score"] = MetricResult(
            score=hal_entry.get("score", 2),
            confidence=hal_entry.get("confidence", "medium"),
            reasoning=hal_entry.get("reasoning", ""),
        )

    if "fih_detected" in selected_metrics:
        fih_list = run.get("fih_detected", [])
        result["fih_detected"] = [
            FIHItem(
                claim=fih.get("claim", ""),
                source_says=fih.get("source_says", ""),
                severity=fih.get("severity", "minor"),
                location=fih.get("location", ""),
            )
            for fih in fih_list
        ]

    return result


# ---------------------------------------------------------------------------
# Section-wise evaluation pipeline
# ---------------------------------------------------------------------------


async def run_section_evaluation(
    request: SectionEvaluationRequest,
) -> SectionEvaluationResponse:
    """Run section-wise evaluation pipeline.

    Flow: resolve input → per-section retrieval → per-section LLM evaluation → aggregate.
    """
    evaluation_id = str(uuid.uuid4())
    start_time = time.monotonic()
    selected_metrics = list(request.metrics)

    # Step 1: Resolve input to normalized ReportJSON
    report: ReportJSON = await resolve_to_json(request)

    logger.info(
        "section_evaluation_start",
        report_id=report.report_id,
        evaluation_id=evaluation_id,
        num_sections=len(report.sections),
        metrics=selected_metrics,
    )

    # Step 2: Evaluate each section independently
    section_results: list[SectionScore] = []
    section_metric_dicts: list[dict] = []

    for section in report.sections:
        section_dict = section.model_dump()

        # Section-specific retrieval
        retrieved_chunks = await retrieve_for_section(
            section_dict,
            top_k=5,
            disease_context=request.disease_context,
            guideline_topic=request.guideline_topic,
        )

        if not retrieved_chunks:
            logger.warning(
                "section_no_chunks",
                section_id=section.id,
                section_title=section.title,
            )
            # Still evaluate — the LLM should note lack of evidence
            # Use empty chunks rather than skipping

        # Build section-level prompts
        system_prompt = build_section_system_prompt(selected_metrics)
        formatted = format_chunks(retrieved_chunks)
        user_prompt = build_section_user_prompt(
            section_title=section.title,
            section_type=section.section_type,
            section_content=section.content,
            formatted_chunks=formatted,
            guideline_topic=request.guideline_topic,
            disease_context=request.disease_context,
        )

        # Call LLM judge for this section
        try:
            raw_result = await call_llm_judge(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                deployment=request.evaluation_model,
                report_id=report.report_id,
                run_index=section.order,
            )
        except Exception as e:
            logger.error(
                "section_evaluation_failed",
                section_id=section.id,
                error=str(e),
            )
            raise RuntimeError(
                f"Section evaluation failed for section '{section.title}': {e}"
            )

        # Extract typed metrics for this section
        metrics_dict = _extract_metrics(raw_result, selected_metrics)

        # Build SectionScore
        section_score = SectionScore(
            section_id=section.id,
            section_title=section.title,
            section_type=section.section_type,
            clinical_accuracy=metrics_dict.get("clinical_accuracy"),
            completeness=metrics_dict.get("completeness"),
            safety_completeness=metrics_dict.get("safety_completeness"),
            relevance=metrics_dict.get("relevance"),
            coherence=metrics_dict.get("coherence"),
            evidence_traceability=metrics_dict.get("evidence_traceability"),
            hallucination_score=metrics_dict.get("hallucination_score"),
            fih_detected=metrics_dict.get("fih_detected"),
            flags=generate_flags(
                safety=metrics_dict.get("safety_completeness"),
                traceability=metrics_dict.get("evidence_traceability"),
                hallucination=metrics_dict.get("hallucination_score"),
                fih_detected=metrics_dict.get("fih_detected"),
                clinical_accuracy=metrics_dict.get("clinical_accuracy"),
            ),
        )
        section_results.append(section_score)

        # Keep dict for aggregation, with content length for optional weighting
        metrics_dict["_content_length"] = len(section.content)
        section_metric_dicts.append(metrics_dict)

        logger.info(
            "section_evaluated",
            section_id=section.id,
            section_title=section.title,
        )

    # Step 3: Aggregate scores across sections
    final_scores = aggregate_section_scores(
        section_metric_dicts, selected_metrics
    )

    # Collect all flags from all sections (deduplicated)
    all_flags = list(
        dict.fromkeys(
            flag for ss in section_results for flag in ss.flags
        )
    )

    # Derive confidence from all per-section metric results
    all_metric_results = [
        getattr(ss, metric)
        for ss in section_results
        for metric in selected_metrics
        if getattr(ss, metric, None) is not None and hasattr(getattr(ss, metric), "confidence")
    ]
    confidence = aggregate_confidence_level(all_metric_results)

    timestamp = datetime.now(timezone.utc).isoformat()

    response = SectionEvaluationResponse(
        report_id=report.report_id,
        evaluation_id=evaluation_id,
        timestamp=timestamp,
        evaluation_model=request.evaluation_model,
        metrics_evaluated=selected_metrics,
        final_scores=final_scores,
        section_scores=section_results,
        confidence_level=confidence,
        flags=all_flags,
        cosmos_document_id=evaluation_id,
    )

    # Store in Cosmos DB
    cosmos_doc = response.model_dump()
    cosmos_doc["id"] = evaluation_id
    cosmos_doc["report_id"] = report.report_id

    try:
        await cosmos_service.store_evaluation(cosmos_doc)
    except Exception as e:
        logger.error(
            "cosmos_store_failed",
            evaluation_id=evaluation_id,
            error=str(e),
        )

    # Store in Blob Storage
    try:
        blob_url = await blob_service.store_evaluation_report(
            report_id=report.report_id,
            evaluation_id=evaluation_id,
            data=cosmos_doc,
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
        "section_evaluation_complete",
        report_id=report.report_id,
        evaluation_id=evaluation_id,
        num_sections=len(section_results),
        elapsed_seconds=elapsed,
    )

    return response
