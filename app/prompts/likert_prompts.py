"""Prompt builders for the Likert evaluation pipeline (1-4 scale metrics)."""

from app.models.metrics import MetricDefinition
from app.models.requests import SourceChunk
from app.prompts.claim_prompts import format_chunks


_LIKERT_SYSTEM = """\
You are an expert clinical guideline evaluator. Your task is to evaluate a \
Clinical Practice Guideline (CPG) report on the following dimension:

## {metric_name}: {metric_description}

Rate each of the following statements using this scale:
- 1 = Strongly Disagree
- 2 = Disagree
- 3 = Agree
- 4 = Strongly Agree

### Statements to Evaluate

{sub_questions_text}

Instructions:
- Read the CPG report and the source evidence chunks carefully.
- For each statement, provide a score (1-4) and brief reasoning (1-2 sentences).
- Be objective and evidence-based in your assessment.
- Also provide an overall reasoning summarizing your evaluation of this dimension.

Respond ONLY with valid JSON (no markdown fences, no preamble). Use this schema:

{{
  "sub_question_scores": [
    {{"sub_question_id": "<id>", "score": <1-4>, "reasoning": "<brief>"}}
  ],
  "overall_reasoning": "<1-3 sentences summarizing this dimension>"
}}"""

_LIKERT_USER = """\
## Guideline Topic
{guideline_topic}

## Disease Context
{disease_context}

## Source Evidence Chunks
{formatted_chunks}

## Generated CPG Report
{generated_report}

Evaluate the CPG report on the dimension described in your instructions. \
Return the JSON result."""


def build_likert_prompt(
    metric: MetricDefinition,
    generated_report: str,
    evidence_chunks: list[SourceChunk],
    guideline_topic: str = "",
    disease_context: str = "",
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for Likert-scale evaluation."""
    sub_questions_text = "\n".join(
        f"{i}. **[{sq.id}]** {sq.text}"
        for i, sq in enumerate(metric.sub_questions, 1)
    )

    system = _LIKERT_SYSTEM.format(
        metric_name=metric.name.replace("_", " ").title(),
        metric_description=metric.description,
        sub_questions_text=sub_questions_text,
    )

    formatted = format_chunks(evidence_chunks) if evidence_chunks else "No evidence chunks available."

    user = _LIKERT_USER.format(
        guideline_topic=guideline_topic,
        disease_context=disease_context,
        formatted_chunks=formatted,
        generated_report=generated_report,
    )

    return system, user
