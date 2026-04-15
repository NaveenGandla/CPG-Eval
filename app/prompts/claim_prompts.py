"""Prompt builders for the claim-level verification pipeline (percentage metrics)."""

from app.models.metrics import SubQuestion
from app.models.requests import SourceChunk


def format_chunks(chunks: list[SourceChunk]) -> str:
    """Format source chunks for inclusion in prompts."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata
        source_parts = []
        if meta.study_name:
            source_parts.append(meta.study_name)
        if meta.year:
            source_parts.append(str(meta.year))
        if meta.journal:
            source_parts.append(meta.journal)
        source_label = ", ".join(source_parts) if source_parts else "Unknown source"
        parts.append(f"[Chunk {i} | id={chunk.chunk_id}] (Source: {source_label})\n{chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 1: Claim extraction
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an expert clinical guideline analyst. Your task is to extract specific \
claims from a Clinical Practice Guideline (CPG) report that are relevant to the \
following evaluation question:

**{sub_question_text}**

Instructions:
- Read the entire CPG report carefully.
- Extract EVERY claim, statement, or mention in the report that is relevant to \
the evaluation question above.
- A "claim" is a specific factual assertion — e.g., a drug dosage, a diagnostic \
threshold, a cited reference, a statistical figure, etc.
- For each claim, provide the exact text from the report and its approximate location.
- If there are NO relevant claims in the report, return an empty array.
- Do NOT fabricate claims — only extract what is actually in the report.

Respond ONLY with valid JSON (no markdown fences, no preamble). Use this schema:

[
  {{"claim_id": "c1", "claim_text": "<exact text from report>", "location": "<section/paragraph>"}}
]"""

_EXTRACTION_SYSTEM_CONSISTENCY = """\
You are an expert clinical guideline analyst. Your task is to extract specific \
statements from a Clinical Practice Guideline (CPG) report that need to be \
checked for internal consistency, related to the following question:

**{sub_question_text}**

Instructions:
- Read the entire CPG report carefully.
- Extract EVERY statement or claim that is relevant to the consistency question above.
- Focus on statements that appear in DIFFERENT parts of the document and could \
potentially contradict each other.
- For each statement, provide the exact text and its location in the document.
- If there are NO relevant statements, return an empty array.

Respond ONLY with valid JSON (no markdown fences, no preamble). Use this schema:

[
  {{"claim_id": "c1", "claim_text": "<exact text from report>", "location": "<section/paragraph>"}}
]"""

_EXTRACTION_USER = """\
## CPG Report

{generated_report}

Extract all claims relevant to: "{sub_question_text}"
Return the JSON array."""


def build_claim_extraction_prompt(
    sub_question: SubQuestion,
    generated_report: str,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for claim extraction."""
    if sub_question.requires_index:
        system = _EXTRACTION_SYSTEM.format(sub_question_text=sub_question.text)
    else:
        system = _EXTRACTION_SYSTEM_CONSISTENCY.format(sub_question_text=sub_question.text)

    user = _EXTRACTION_USER.format(
        generated_report=generated_report,
        sub_question_text=sub_question.text,
    )
    return system, user


# ---------------------------------------------------------------------------
# Step 3a: Claim verification against evidence (index-based)
# ---------------------------------------------------------------------------

_VERIFICATION_SYSTEM = """\
You are an expert clinical guideline evaluator. Your task is to verify whether \
claims from a CPG report are correct based on the provided source evidence.

Evaluation question: **{sub_question_text}**

For each claim, determine:
- "correct": The claim is factually accurate and supported by the evidence.
- "incorrect": The claim contradicts the evidence or contains factual errors.
- "unverifiable": The evidence provided is insufficient to confirm or deny the claim.

Instructions:
- Compare each claim against the source evidence chunks provided.
- Be precise — a minor numerical difference (e.g., rounding) is still "correct" \
if it does not change clinical meaning.
- A claim that is partially correct but contains a clinically meaningful error \
should be marked "incorrect".
- Provide brief reasoning for each verdict.
- Reference the evidence chunk ID that supports your verdict when applicable.

Respond ONLY with valid JSON (no markdown fences, no preamble). Use this schema:

[
  {{"claim_id": "<id>", "verdict": "correct|incorrect|unverifiable", "reasoning": "<brief>", "evidence_chunk_id": "<chunk_id or null>"}}
]"""

_VERIFICATION_USER = """\
## Claims to Verify

{claims_text}

## Source Evidence Chunks

{evidence_text}

Verify each claim against the evidence. Return the JSON array."""


def build_claim_verification_prompt(
    claims: list[dict],
    evidence_chunks: list[SourceChunk],
    sub_question: SubQuestion,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for index-based claim verification."""
    system = _VERIFICATION_SYSTEM.format(sub_question_text=sub_question.text)

    claims_text = "\n".join(
        f"- [{c['claim_id']}] \"{c['claim_text']}\" (Location: {c['location']})"
        for c in claims
    )
    evidence_text = format_chunks(evidence_chunks) if evidence_chunks else "No evidence chunks available."

    user = _VERIFICATION_USER.format(
        claims_text=claims_text,
        evidence_text=evidence_text,
    )
    return system, user


# ---------------------------------------------------------------------------
# Step 3b: Consistency verification (self-comparison)
# ---------------------------------------------------------------------------

_CONSISTENCY_SYSTEM = """\
You are an expert clinical guideline evaluator. Your task is to check whether \
statements in a CPG report are internally consistent with each other and with \
the rest of the document.

Consistency question: **{sub_question_text}**

For each statement, determine:
- "correct": The statement is consistent with the rest of the document — no contradictions found.
- "incorrect": The statement contradicts another part of the document.
- "unverifiable": Cannot determine consistency from the document alone.

Instructions:
- Read the full CPG report provided below.
- For each extracted statement, check if it is consistent with all other related \
statements in the document.
- If you find a contradiction, specify the conflicting location in the document.
- Provide brief reasoning for each verdict.

Respond ONLY with valid JSON (no markdown fences, no preamble). Use this schema:

[
  {{"claim_id": "<id>", "verdict": "correct|incorrect|unverifiable", "reasoning": "<brief>", "conflicting_location": "<location or null>"}}
]"""

_CONSISTENCY_USER = """\
## Statements to Check for Consistency

{claims_text}

## Full CPG Report

{generated_report}

Check each statement for internal consistency with the rest of the document. Return the JSON array."""


def build_consistency_verification_prompt(
    claims: list[dict],
    generated_report: str,
    sub_question: SubQuestion,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for consistency verification."""
    system = _CONSISTENCY_SYSTEM.format(sub_question_text=sub_question.text)

    claims_text = "\n".join(
        f"- [{c['claim_id']}] \"{c['claim_text']}\" (Location: {c['location']})"
        for c in claims
    )

    user = _CONSISTENCY_USER.format(
        claims_text=claims_text,
        generated_report=generated_report,
    )
    return system, user
