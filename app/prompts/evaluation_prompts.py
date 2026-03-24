from app.models.requests import SourceChunk

# Individual dimension prompt blocks keyed by metric name
DIMENSION_PROMPTS: dict[str, str] = {
    "clinical_accuracy": """\
### Clinical Accuracy (1-5 Likert)
Are drug names, dosages, trial names, statistical outcomes (OS, PFS, ORR), and clinical conclusions factually correct when compared to the source evidence?
- 5: All clinical facts are correct and precisely stated
- 4: Minor inaccuracies that do not change clinical meaning (e.g., rounding differences)
- 3: Some inaccuracies present but core conclusions remain valid
- 2: Multiple factual errors that could mislead clinical interpretation
- 1: Pervasive inaccuracies rendering the report unreliable""",
    "completeness": """\
### Completeness (1-5 Likert)
Does the report cover all critical aspects present in the source evidence: efficacy data, key trials, patient populations, comparator arms, and endpoints?
- 5: All critical information from source evidence is included
- 4: Most critical information included, minor gaps only
- 3: Several important details missing but main findings covered
- 2: Significant gaps in coverage of source evidence
- 1: Majority of critical information is missing""",
    "safety_completeness": """\
### Safety Completeness (1-5 Likert)
Are adverse effects, contraindications, drug interactions, dose modifications, black box warnings, and toxicity profiles adequately covered? This is evaluated SEPARATELY because LLMs systematically under-report safety data.
- 5: Comprehensive safety profile with specific AE rates and grades
- 4: Most safety information present, minor omissions
- 3: Safety mentioned but lacks specificity (e.g., "well-tolerated" without data)
- 2: Safety information is superficial or significantly incomplete
- 1: Safety information is absent or dangerously incomplete""",
    "relevance": """\
### Relevance (1-5 Likert)
Is all included information directly pertinent to the guideline topic? No off-topic, tangential, or outdated content that does not serve the clinical question?
- 5: Every piece of information directly serves the clinical question
- 4: Mostly relevant with minor tangential content
- 3: Some irrelevant information included alongside relevant content
- 2: Significant amount of off-topic or outdated information
- 1: Mostly irrelevant to the guideline topic""",
    "coherence": """\
### Coherence (1-5 Likert)
Is the report logically structured with clear sections, smooth transitions, and a clinical reasoning flow that a healthcare professional can follow?
- 5: Excellent structure, reads as a professional clinical document
- 4: Well-organized with minor structural issues
- 3: Adequate structure but some sections feel disjointed
- 2: Poor organization that hinders comprehension
- 1: Incoherent, no logical flow""",
    "evidence_traceability": """\
### Evidence Traceability (1-5 Likert)
Can every factual claim in the report be traced back to a specific source chunk provided in the retrieved evidence? Are citations present and correct?
- 5: Every claim is traceable to a specific source with correct attribution
- 4: Most claims traceable, few minor attribution gaps
- 3: Some claims lack source attribution but appear grounded
- 2: Many claims cannot be traced to provided sources
- 1: Most claims are unattributed or attributed to wrong sources""",
    "hallucination_score": """\
### Hallucination Score (1-4 ordinal)
Does the report contain fabricated facts, non-existent studies, invented statistics, or claims that contradict the source evidence? No neutral midpoint — a claim is either grounded or it is not.
- 4: No hallucinations detected — all content is grounded in source evidence
- 3: Few minor hallucinations (e.g., slightly off statistics) that do not affect clinical conclusions
- 2: Some hallucinations present, including potentially misleading claims
- 1: Many hallucinations — fabricated studies, invented data, or contradictory claims""",
    "fih_detected": """\
### Factually Incorrect Hallucinations (FIH Detection)
For this dimension, identify SPECIFIC claims in the report that are factually incorrect — meaning they contradict the source evidence OR contradict established medical knowledge. For each detected FIH, provide:
- The exact claim text from the report
- What the source evidence actually says (or "not found in sources" if the claim has no basis)
- Severity: "critical" (could harm patient outcomes), "major" (significantly misleading), "minor" (unlikely to affect clinical decisions)
- Location: paragraph number or section where the claim appears""",
}

# JSON schema fragments for each metric in the output format
OUTPUT_SCHEMAS: dict[str, str] = {
    "clinical_accuracy": '  "clinical_accuracy": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>"}',
    "completeness": '  "completeness": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>"}',
    "safety_completeness": '  "safety_completeness": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>", "missing_items": ["<string>"]}',
    "relevance": '  "relevance": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>"}',
    "coherence": '  "coherence": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>"}',
    "evidence_traceability": '  "evidence_traceability": {"score": <1-5>, "confidence": "<high|medium|low>", "reasoning": "<string>", "untraced_claims": [{"claim": "<string>", "location": "<string>"}]}',
    "hallucination_score": '  "hallucination_score": {"score": <1-4>, "confidence": "<high|medium|low>", "reasoning": "<string>"}',
    "fih_detected": '  "fih_detected": [{"claim": "<string>", "source_says": "<string>", "severity": "<critical|major|minor>", "location": "<string>"}]',
}


def build_system_prompt(metrics: list[str]) -> str:
    """Build the system prompt dynamically based on selected metrics."""
    num_dimensions = len(metrics)

    dimension_sections = []
    for i, metric in enumerate(metrics, 1):
        section = DIMENSION_PROMPTS[metric]
        # Prefix with dimension number
        lines = section.split("\n")
        lines[0] = f"### {i}. " + lines[0].split("### ", 1)[1]
        dimension_sections.append("\n".join(lines))

    dimensions_text = "\n\n".join(dimension_sections)

    # Build evaluation process steps
    scored_metrics = [m for m in metrics if m != "fih_detected"]
    has_fih = "fih_detected" in metrics

    process_steps = [
        "1. Read the guideline topic and disease context to understand the clinical scope",
        "2. Read ALL source evidence chunks carefully — these are your ground truth",
        "3. Read the generated CPG report end-to-end",
    ]
    if scored_metrics:
        process_steps.append(
            f"4. For each scored dimension, compare the report against source evidence and assign a score with reasoning"
        )
    if has_fih:
        process_steps.append(
            f"{len(process_steps) + 1}. For FIH detection, go through the report claim-by-claim and cross-reference with source chunks"
        )
        process_steps.append(
            f"{len(process_steps) + 1}. If a claim in the report is NOT found in any source chunk, flag it as an FIH only if it is factually wrong (not merely unsupported but correct domain knowledge)"
        )
    process_steps.append(
        f"{len(process_steps) + 1}. Prioritize clinical accuracy and safety over writing quality — a well-written but inaccurate report should score LOW overall"
    )

    process_text = "\n".join(process_steps)

    # Build output schema
    schema_lines = [OUTPUT_SCHEMAS[m] for m in metrics]
    schema_text = ",\n".join(schema_lines)

    return f"""\
You are an expert clinical guideline evaluator. Your task is to evaluate an LLM-generated Clinical Practice Guideline (CPG) report against the source evidence chunks that were retrieved to generate it.

You must evaluate the report across {num_dimensions} dimension{"s" if num_dimensions > 1 else ""}. For each scored dimension, provide:
1. A numeric score on the specified scale
2. A confidence level (high/medium/low)
3. A brief reasoning (2-3 sentences max)

## Evaluation Dimensions

{dimensions_text}

## Evaluation Process (Chain-of-Thought)

Follow these steps in order:
{process_text}

## Output Format

Respond ONLY with valid JSON (no markdown, no preamble, no explanation outside the JSON). Use this exact schema:

{{
{schema_text}
}}"""


def format_chunks(chunks: list[SourceChunk]) -> str:
    """Format source chunks for the user prompt."""
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
        parts.append(f"[Chunk {i}] (Source: {source_label})\n{chunk.text}")
    return "\n\n".join(parts)


def build_user_prompt(
    guideline_topic: str,
    disease_context: str,
    formatted_chunks: str,
    generated_report: str,
) -> str:
    """Build the user prompt for the LLM judge."""
    return f"""\
## Guideline Topic
{guideline_topic}

## Disease Context
{disease_context}

## Source Evidence Chunks
{formatted_chunks}

## Generated CPG Report
{generated_report}

Evaluate the above CPG report against the provided source evidence chunks. Follow the evaluation process described in your instructions and return the JSON result."""


# ---------------------------------------------------------------------------
# Section-level prompt builders
# ---------------------------------------------------------------------------


def build_section_system_prompt(metrics: list[str]) -> str:
    """Build system prompt for evaluating a SINGLE SECTION of a CPG report."""
    num_dimensions = len(metrics)

    dimension_sections = []
    for i, metric in enumerate(metrics, 1):
        section = DIMENSION_PROMPTS[metric]
        lines = section.split("\n")
        lines[0] = f"### {i}. " + lines[0].split("### ", 1)[1]
        dimension_sections.append("\n".join(lines))

    dimensions_text = "\n\n".join(dimension_sections)

    scored_metrics = [m for m in metrics if m != "fih_detected"]
    has_fih = "fih_detected" in metrics

    process_steps = [
        "1. Read the section title, type, and content carefully",
        "2. Read ALL retrieved evidence chunks — these are your ground truth for THIS section",
        "3. Evaluate ONLY the content within this section",
    ]
    if scored_metrics:
        process_steps.append(
            "4. For each scored dimension, compare the section content against retrieved evidence and assign a score with reasoning"
        )
    if has_fih:
        process_steps.append(
            f"{len(process_steps) + 1}. For FIH detection, go through the section claim-by-claim and cross-reference with source chunks"
        )
        process_steps.append(
            f"{len(process_steps) + 1}. If a claim in the section is NOT found in any source chunk, flag it as an FIH only if it is factually wrong"
        )
    process_steps.append(
        f"{len(process_steps) + 1}. Prioritize clinical accuracy and safety over writing quality"
    )

    process_text = "\n".join(process_steps)

    schema_lines = [OUTPUT_SCHEMAS[m] for m in metrics]
    schema_text = ",\n".join(schema_lines)

    return f"""\
You are evaluating a SINGLE SECTION of a Clinical Practice Guideline (CPG) report. Your task is to evaluate this section against the retrieved evidence chunks that are specific to this section's topic.

IMPORTANT CONSTRAINTS:
- You are evaluating ONLY this section, not the full document.
- Do NOT use assumptions from other sections.
- Base your evaluation ONLY on the evidence chunks provided for this section.

You must evaluate the section across {num_dimensions} dimension{"s" if num_dimensions > 1 else ""}. For each scored dimension, provide:
1. A numeric score on the specified scale
2. A confidence level (high/medium/low)
3. A brief reasoning (2-3 sentences max)

## Evaluation Dimensions

{dimensions_text}

## Evaluation Process (Chain-of-Thought)

Follow these steps in order:
{process_text}

## Output Format

Respond ONLY with valid JSON (no markdown, no preamble, no explanation outside the JSON). Use this exact schema:

{{
{schema_text}
}}"""


def build_section_user_prompt(
    section_title: str,
    section_type: str,
    section_content: str,
    formatted_chunks: str,
    guideline_topic: str,
    disease_context: str,
) -> str:
    """Build user prompt for evaluating a single section."""
    return f"""\
## Guideline Topic
{guideline_topic}

## Disease Context
{disease_context}

## Section Being Evaluated
**Title:** {section_title}
**Type:** {section_type}

### Section Content
{section_content}

## Retrieved Evidence Chunks (for this section)
{formatted_chunks}

Evaluate the above SECTION of a CPG report against the provided evidence chunks. Focus ONLY on this section's content. Do NOT make assumptions about content in other sections. Return the JSON result."""
