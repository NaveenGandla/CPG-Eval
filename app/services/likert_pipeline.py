"""Likert-scale evaluation pipeline for 1-4 scale metrics."""

import structlog

from app.models.metrics import MetricDefinition
from app.models.requests import SourceChunk
from app.models.responses import LikertMetricResult, LikertSubQuestionScore
from app.prompts.likert_prompts import build_likert_prompt
from app.services.llm_judge import call_llm_judge

logger = structlog.get_logger()


async def run_likert_metric(
    metric: MetricDefinition,
    generated_report: str,
    evidence_chunks: list[SourceChunk],
    deployment: str,
    report_id: str,
    guideline_topic: str = "",
    disease_context: str = "",
) -> LikertMetricResult:
    """Evaluate a Likert metric with a single LLM call scoring all sub-questions."""
    system_prompt, user_prompt = build_likert_prompt(
        metric=metric,
        generated_report=generated_report,
        evidence_chunks=evidence_chunks,
        guideline_topic=guideline_topic,
        disease_context=disease_context,
    )

    raw = await call_llm_judge(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        deployment=deployment,
        report_id=report_id,
    )

    # Parse sub-question scores
    raw_scores = raw.get("sub_question_scores", [])
    overall_reasoning = raw.get("overall_reasoning", "")

    # Build a lookup from sub-question ID to its definition
    sq_lookup = {sq.id: sq for sq in metric.sub_questions}

    sub_scores: list[LikertSubQuestionScore] = []
    for item in raw_scores:
        sq_id = item.get("sub_question_id", "")
        score = item.get("score", 2)
        # Clamp to valid range
        score = max(1, min(4, score))
        reasoning = item.get("reasoning", "")

        sq_text = sq_lookup[sq_id].text if sq_id in sq_lookup else sq_id

        sub_scores.append(
            LikertSubQuestionScore(
                sub_question_id=sq_id,
                sub_question_text=sq_text,
                score=score,
                reasoning=reasoning,
            )
        )

    # Fill in any missing sub-questions with default score of 2
    scored_ids = {s.sub_question_id for s in sub_scores}
    for sq in metric.sub_questions:
        if sq.id not in scored_ids:
            logger.warning(
                "likert_missing_sub_question",
                metric=metric.name,
                sub_question_id=sq.id,
            )
            sub_scores.append(
                LikertSubQuestionScore(
                    sub_question_id=sq.id,
                    sub_question_text=sq.text,
                    score=2,
                    reasoning="Score not returned by evaluator.",
                )
            )

    # Average scores
    if sub_scores:
        avg = sum(s.score for s in sub_scores) / len(sub_scores)
    else:
        avg = 2.0

    return LikertMetricResult(
        score=round(avg, 2),
        sub_questions=sub_scores,
        overall_reasoning=overall_reasoning,
    )
