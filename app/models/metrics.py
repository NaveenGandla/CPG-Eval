"""Metric definitions and sub-question registry for CPG evaluation."""

from dataclasses import dataclass, field
from enum import Enum


class MetricType(Enum):
    PERCENTAGE = "percentage"
    LIKERT = "likert"


@dataclass(frozen=True)
class SubQuestion:
    id: str
    text: str
    requires_index: bool = True  # False for self-comparison (Consistency)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    metric_type: MetricType
    description: str
    sub_questions: tuple[SubQuestion, ...] = field(default_factory=tuple)


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    "accuracy": MetricDefinition(
        name="accuracy",
        metric_type=MetricType.PERCENTAGE,
        description=(
            "All clinical facts, criteria, numbers, dosages, and recommendations "
            "in the document are factually correct based on the cited source "
            "literature/lab reference ranges."
        ),
        sub_questions=(
            SubQuestion(
                id="accuracy_diagnostic_criteria",
                text="Are diagnostic criteria, classification thresholds, and pathways accurate?",
            ),
            SubQuestion(
                id="accuracy_lab_ranges",
                text="Are laboratory reference ranges and monitoring parameters correct?",
            ),
            SubQuestion(
                id="accuracy_drug_dosages",
                text="Are drug dosages, frequencies, and routes of administration correct?",
            ),
            SubQuestion(
                id="accuracy_drug_interactions",
                text="Are drug interaction and contraindication claims accurate?",
            ),
        ),
    ),
    "hallucinations": MetricDefinition(
        name="hallucinations",
        metric_type=MetricType.PERCENTAGE,
        description=(
            "The document is free from fabricated facts, citations, statistics, "
            "dosages, or recommendations that cannot be traced to any known or "
            "cited source material."
        ),
        sub_questions=(
            SubQuestion(
                id="hallucination_references",
                text="Are all cited references real and verifiable publications?",
            ),
            SubQuestion(
                id="hallucination_statistics",
                text="Are statistics and numerical claims attributable to a known source?",
            ),
            SubQuestion(
                id="hallucination_recommendations",
                text="Does every recommendation map to a cited guideline or approved sources?",
            ),
            SubQuestion(
                id="hallucination_treatments",
                text="Are any treatment recommendations stated that do not appear in any cited guideline?",
            ),
            SubQuestion(
                id="hallucination_entities",
                text="Are any clinical entities (drug names, scoring systems, classifications) fabricated or non-existent?",
            ),
            SubQuestion(
                id="hallucination_fake_citations",
                text="Are fake citations or blended guidelines created?",
            ),
            SubQuestion(
                id="hallucination_invented_thresholds",
                text="Does it invent thresholds, drug doses, and risk cut offs?",
            ),
        ),
    ),
    "consistency": MetricDefinition(
        name="consistency",
        metric_type=MetricType.PERCENTAGE,
        description=(
            "The document is free from contradictions between different sections "
            "(e.g., diagnostic criteria vs. treatment recommendations, or "
            "conflicting dosage guidance)."
        ),
        sub_questions=(
            SubQuestion(
                id="consistency_diagnostic_treatment",
                text="Are diagnostic criteria consistent with the treatment algorithm?",
                requires_index=False,
            ),
            SubQuestion(
                id="consistency_dosages",
                text="Are dosage recommendations consistent across all sections where they appear?",
                requires_index=False,
            ),
            SubQuestion(
                id="consistency_summary_body",
                text="Do the summary / key recommendations align with the detailed body text?",
                requires_index=False,
            ),
            SubQuestion(
                id="consistency_referral_severity",
                text="Are referral and escalation criteria consistent with the stated severity classifications?",
                requires_index=False,
            ),
            SubQuestion(
                id="consistency_identical_inputs",
                text="Does the model give different recommendations for identical inputs?",
                requires_index=False,
            ),
            SubQuestion(
                id="consistency_cpg_pathways",
                text="Is the content consistent between the CPG and the clinical pathways/protocols?",
                requires_index=False,
            ),
        ),
    ),
    "source_traceability": MetricDefinition(
        name="source_traceability",
        metric_type=MetricType.PERCENTAGE,
        description=(
            "Each key recommendation or clinical claim in the document can be "
            "traced back to a specific cited reference."
        ),
        sub_questions=(
            SubQuestion(
                id="traceability_claims",
                text="Each key recommendation or clinical claim in the document can be traced back to a specific cited reference.",
            ),
        ),
    ),
    "coherence": MetricDefinition(
        name="coherence",
        metric_type=MetricType.LIKERT,
        description=(
            "The document is logically structured and reads as a unified, "
            "coherent guideline."
        ),
        sub_questions=(
            SubQuestion(
                id="coherence_pathway_alignment",
                text="The clinical pathway aligns with the recommendations in the guideline.",
            ),
            SubQuestion(
                id="coherence_sections",
                text="Different sections of the document support each other coherently.",
            ),
            SubQuestion(
                id="coherence_terminology",
                text="The terminology used is consistent throughout the document.",
            ),
            SubQuestion(
                id="coherence_unified",
                text="The document reads as a unified, coherent guideline rather than fragmented outputs.",
            ),
        ),
    ),
    "clinical_relevance": MetricDefinition(
        name="clinical_relevance",
        metric_type=MetricType.LIKERT,
        description=(
            "The recommendations are clinically appropriate and reflect "
            "current best practices."
        ),
        sub_questions=(
            SubQuestion(
                id="relevance_appropriate",
                text="The recommendations are clinically appropriate.",
            ),
            SubQuestion(
                id="relevance_best_practices",
                text="The guideline reflects current best practices.",
            ),
            SubQuestion(
                id="relevance_suited",
                text="The pathway is well suited for relevant clinical practice.",
            ),
            SubQuestion(
                id="relevance_applicable",
                text="The recommendations are applicable to my current clinical practices.",
            ),
        ),
    ),
    "bias": MetricDefinition(
        name="bias",
        metric_type=MetricType.LIKERT,
        description=(
            "The recommendations are free from demographic, commercial, "
            "or selection bias."
        ),
        sub_questions=(
            SubQuestion(
                id="bias_demographic",
                text="The recommendations are free from demographic, commercial or selection bias.",
            ),
            SubQuestion(
                id="bias_guideline_priority",
                text="The Agent does not systematically prioritize one guideline body without justification.",
            ),
            SubQuestion(
                id="bias_non_pharma",
                text="Non-pharmacological interventions are adequately represented.",
            ),
            SubQuestion(
                id="bias_conflicting_info",
                text="The agent does not exclude conflicting information or downplay it.",
            ),
        ),
    ),
    "transparency": MetricDefinition(
        name="transparency",
        metric_type=MetricType.LIKERT,
        description=(
            "The cited sources support each recommendation and the rationale "
            "is clear and consistent."
        ),
        sub_questions=(
            SubQuestion(
                id="transparency_sources",
                text="The cited sources support each recommendation.",
            ),
            SubQuestion(
                id="transparency_traceable",
                text="I can clearly trace recommendations back to guideline documents.",
            ),
            SubQuestion(
                id="transparency_citation_detail",
                text="The level of citation detail is sufficient for verification.",
            ),
            SubQuestion(
                id="transparency_rationale",
                text="The rationale provided for recommendations is clear and consistent.",
            ),
        ),
    ),
}

PERCENTAGE_METRICS = [
    name for name, m in METRIC_REGISTRY.items()
    if m.metric_type == MetricType.PERCENTAGE
]

LIKERT_METRICS = [
    name for name, m in METRIC_REGISTRY.items()
    if m.metric_type == MetricType.LIKERT
]
