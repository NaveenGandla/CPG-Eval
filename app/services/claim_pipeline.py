"""Claim-level verification pipeline for percentage-based metrics."""

import asyncio

import structlog

from app.config import settings
from app.models.metrics import MetricDefinition, SubQuestion
from app.models.responses import (
    ClaimVerdict,
    ExtractedClaim,
    PercentageMetricResult,
    SubQuestionResult,
)
from app.prompts.claim_prompts import (
    build_claim_extraction_prompt,
    build_claim_verification_prompt,
    build_consistency_verification_prompt,
)
from app.services.llm_judge import call_llm_judge_list
from app.services.search_service import retrieve_for_claim

logger = structlog.get_logger()


async def run_percentage_metric(
    metric: MetricDefinition,
    generated_report: str,
    guideline_topic: str,
    disease_context: str,
    deployment: str,
    report_id: str,
) -> PercentageMetricResult:
    """Evaluate a percentage metric by running all sub-questions in parallel."""
    tasks = [
        _evaluate_sub_question(
            sub_question=sq,
            generated_report=generated_report,
            guideline_topic=guideline_topic,
            disease_context=disease_context,
            deployment=deployment,
            report_id=report_id,
        )
        for sq in metric.sub_questions
    ]

    sub_results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results: list[SubQuestionResult] = []
    for i, result in enumerate(sub_results):
        if isinstance(result, Exception):
            logger.error(
                "sub_question_failed",
                metric=metric.name,
                sub_question_id=metric.sub_questions[i].id,
                error=str(result),
            )
            # Record a failed sub-question with 100% (no claims to be wrong about)
            valid_results.append(
                SubQuestionResult(
                    sub_question_id=metric.sub_questions[i].id,
                    sub_question_text=metric.sub_questions[i].text,
                    percentage=100.0,
                )
            )
        else:
            valid_results.append(result)

    # Average sub-question percentages
    if valid_results:
        avg_score = sum(r.percentage for r in valid_results) / len(valid_results)
    else:
        avg_score = 100.0

    return PercentageMetricResult(
        score=round(avg_score, 2),
        sub_questions=valid_results,
    )


async def _evaluate_sub_question(
    sub_question: SubQuestion,
    generated_report: str,
    guideline_topic: str,
    disease_context: str,
    deployment: str,
    report_id: str,
) -> SubQuestionResult:
    """Run the 3-step pipeline for a single sub-question."""
    # Step 1: Extract claims
    claims = await _extract_claims(
        sub_question=sub_question,
        generated_report=generated_report,
        deployment=deployment,
        report_id=report_id,
    )

    if not claims:
        # No claims found — score is 100% (nothing to be wrong about)
        return SubQuestionResult(
            sub_question_id=sub_question.id,
            sub_question_text=sub_question.text,
            claims_extracted=[],
            verifications=[],
            correct_count=0,
            total_count=0,
            percentage=100.0,
        )

    extracted = [
        ExtractedClaim(
            claim_id=c["claim_id"],
            claim_text=c["claim_text"],
            location=c.get("location", ""),
        )
        for c in claims
    ]

    # Step 2 & 3: Retrieve evidence and verify
    if sub_question.requires_index:
        verifications = await _retrieve_and_verify(
            claims=claims,
            sub_question=sub_question,
            guideline_topic=guideline_topic,
            disease_context=disease_context,
            deployment=deployment,
            report_id=report_id,
        )
    else:
        verifications = await _verify_consistency(
            claims=claims,
            generated_report=generated_report,
            sub_question=sub_question,
            deployment=deployment,
            report_id=report_id,
        )

    # Calculate percentage
    correct_count = sum(1 for v in verifications if v.verdict == "correct")
    total_count = len(verifications)
    percentage = (correct_count / total_count * 100) if total_count > 0 else 100.0

    return SubQuestionResult(
        sub_question_id=sub_question.id,
        sub_question_text=sub_question.text,
        claims_extracted=extracted,
        verifications=verifications,
        correct_count=correct_count,
        total_count=total_count,
        percentage=round(percentage, 2),
    )


async def _extract_claims(
    sub_question: SubQuestion,
    generated_report: str,
    deployment: str,
    report_id: str,
) -> list[dict]:
    """Step 1: Extract claims from the report relevant to the sub-question."""
    system_prompt, user_prompt = build_claim_extraction_prompt(
        sub_question=sub_question,
        generated_report=generated_report,
    )

    raw = await call_llm_judge_list(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        deployment=deployment,
        report_id=report_id,
    )

    # Validate and filter
    valid_claims = []
    for item in raw:
        if isinstance(item, dict) and "claim_text" in item:
            if "claim_id" not in item:
                item["claim_id"] = f"c{len(valid_claims) + 1}"
            valid_claims.append(item)

    logger.info(
        "claims_extracted",
        sub_question_id=sub_question.id,
        num_claims=len(valid_claims),
    )
    return valid_claims


async def _retrieve_and_verify(
    claims: list[dict],
    sub_question: SubQuestion,
    guideline_topic: str,
    disease_context: str,
    deployment: str,
    report_id: str,
) -> list[ClaimVerdict]:
    """Step 2+3 for index-based metrics: retrieve evidence per claim, then verify in batches."""
    # Step 2: Parallel retrieval for all claims
    retrieval_tasks = [
        retrieve_for_claim(
            claim_text=c["claim_text"],
            disease_context=disease_context,
            guideline_topic=guideline_topic,
            top_k=settings.percentage_metric_top_k,
        )
        for c in claims
    ]
    all_evidence = await asyncio.gather(*retrieval_tasks)

    # Build evidence map: claim_id -> chunks
    evidence_map: dict[str, list] = {}
    for claim, chunks in zip(claims, all_evidence):
        evidence_map[claim["claim_id"]] = chunks

    # Merge all unique chunks for batch verification
    all_chunks_by_id: dict[str, object] = {}
    for chunks in all_evidence:
        for chunk in chunks:
            all_chunks_by_id[chunk.chunk_id] = chunk
    all_unique_chunks = list(all_chunks_by_id.values())

    # Step 3: Verify in batches
    batch_size = settings.claim_verification_batch_size
    batches = [claims[i:i + batch_size] for i in range(0, len(claims), batch_size)]

    verification_tasks = [
        _verify_batch(
            batch=batch,
            evidence_chunks=all_unique_chunks,
            sub_question=sub_question,
            deployment=deployment,
            report_id=report_id,
        )
        for batch in batches
    ]

    batch_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

    verdicts: list[ClaimVerdict] = []
    for result in batch_results:
        if isinstance(result, Exception):
            logger.error("verification_batch_failed", error=str(result))
        else:
            verdicts.extend(result)

    return verdicts


async def _verify_batch(
    batch: list[dict],
    evidence_chunks: list,
    sub_question: SubQuestion,
    deployment: str,
    report_id: str,
) -> list[ClaimVerdict]:
    """Verify a batch of claims against evidence."""
    system_prompt, user_prompt = build_claim_verification_prompt(
        claims=batch,
        evidence_chunks=evidence_chunks,
        sub_question=sub_question,
    )

    raw = await call_llm_judge_list(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        deployment=deployment,
        report_id=report_id,
    )

    verdicts = []
    for item in raw:
        if isinstance(item, dict) and "verdict" in item:
            verdicts.append(
                ClaimVerdict(
                    claim_id=item.get("claim_id", ""),
                    verdict=item.get("verdict", "unverifiable"),
                    reasoning=item.get("reasoning", ""),
                    evidence_chunk_id=item.get("evidence_chunk_id"),
                )
            )

    return verdicts


async def _verify_consistency(
    claims: list[dict],
    generated_report: str,
    sub_question: SubQuestion,
    deployment: str,
    report_id: str,
) -> list[ClaimVerdict]:
    """Step 3 for consistency metrics: verify claims against the full document."""
    batch_size = settings.claim_verification_batch_size
    batches = [claims[i:i + batch_size] for i in range(0, len(claims), batch_size)]

    tasks = [
        _verify_consistency_batch(
            batch=batch,
            generated_report=generated_report,
            sub_question=sub_question,
            deployment=deployment,
            report_id=report_id,
        )
        for batch in batches
    ]

    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts: list[ClaimVerdict] = []
    for result in batch_results:
        if isinstance(result, Exception):
            logger.error("consistency_batch_failed", error=str(result))
        else:
            verdicts.extend(result)

    return verdicts


async def _verify_consistency_batch(
    batch: list[dict],
    generated_report: str,
    sub_question: SubQuestion,
    deployment: str,
    report_id: str,
) -> list[ClaimVerdict]:
    """Verify a batch of claims for internal consistency."""
    system_prompt, user_prompt = build_consistency_verification_prompt(
        claims=batch,
        generated_report=generated_report,
        sub_question=sub_question,
    )

    raw = await call_llm_judge_list(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        deployment=deployment,
        report_id=report_id,
    )

    verdicts = []
    for item in raw:
        if isinstance(item, dict) and "verdict" in item:
            verdicts.append(
                ClaimVerdict(
                    claim_id=item.get("claim_id", ""),
                    verdict=item.get("verdict", "unverifiable"),
                    reasoning=item.get("reasoning", ""),
                    conflicting_location=item.get("conflicting_location"),
                )
            )

    return verdicts
