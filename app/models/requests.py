from typing import Literal

from pydantic import BaseModel, Field

# All available metric names for multi-select
MetricName = Literal[
    "clinical_accuracy",
    "completeness",
    "safety_completeness",
    "relevance",
    "coherence",
    "evidence_traceability",
    "hallucination_score",
    "fih_detected",
]

ALL_METRICS: list[str] = [
    "clinical_accuracy",
    "completeness",
    "safety_completeness",
    "relevance",
    "coherence",
    "evidence_traceability",
    "hallucination_score",
    "fih_detected",
]


class SourceChunkMetadata(BaseModel):
    study_name: str | None = None
    year: int | None = None
    journal: str | None = None
    authors: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: SourceChunkMetadata = SourceChunkMetadata()


class EvaluationRequest(BaseModel):
    report_id: str = Field(..., description="Unique ID for the CPG report")
    generated_report: str = Field(
        ..., min_length=1, description="Full text of the generated report"
    )
    guideline_topic: str = Field(
        ...,
        min_length=1,
        description="e.g. 'First-line treatment for transplant-eligible NDMM'",
    )
    disease_context: str = Field(
        ..., min_length=1, description="e.g. 'Multiple Myeloma'"
    )
    metrics: list[MetricName] = Field(
        default=ALL_METRICS.copy(),
        min_length=1,
        description=(
            "Metrics to evaluate. Choose from: clinical_accuracy, completeness, "
            "safety_completeness, relevance, coherence, evidence_traceability, "
            "hallucination_score, fih_detected. Defaults to all."
        ),
    )
    reference_report: str | None = Field(
        default=None, description="Optional gold-standard report"
    )
    evaluation_model: str = Field(
        default="gpt-4o", description="Azure OpenAI deployment name for the evaluator"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "report_id": "rpt-001",
                    "generated_report": "This CPG report covers first-line treatment options for newly diagnosed multiple myeloma...",
                    "guideline_topic": "First-line treatment for transplant-eligible NDMM",
                    "disease_context": "Multiple Myeloma",
                    "metrics": [
                        "clinical_accuracy",
                        "safety_completeness",
                        "hallucination_score",
                        "fih_detected",
                    ],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Section-based evaluation models
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A single section from a structured CPG report."""

    id: str = Field(..., description="Unique section identifier")
    title: str = Field(..., description="Section title")
    content: str = Field(..., min_length=1, description="Section text content")
    section_type: str = Field(
        default="general",
        description="Inferred type: definitions, abbreviations, guideline, general",
    )
    order: int = Field(..., description="Order of section in the document")
    keywords: list[str] = Field(
        default_factory=list, description="Top keywords for retrieval"
    )


class ReportJSON(BaseModel):
    """Structured JSON representation of a CPG report."""

    report_id: str
    sections: list[ReportSection] = Field(..., min_length=1)


class SectionEvaluationRequest(BaseModel):
    """Unified evaluation request supporting JSON input, blob path, or file upload."""

    guideline_topic: str = Field(
        ...,
        min_length=1,
        description="e.g. 'First-line treatment for transplant-eligible NDMM'",
    )
    disease_context: str = Field(
        ..., min_length=1, description="e.g. 'Multiple Myeloma'"
    )
    metrics: list[MetricName] = Field(
        default=ALL_METRICS.copy(),
        min_length=1,
        description="Metrics to evaluate. Defaults to all.",
    )
    evaluation_model: str = Field(
        default="gpt-4o", description="Azure OpenAI deployment name"
    )

    # Mode A: JSON input — provide one of these
    report_json: ReportJSON | None = Field(
        default=None, description="Inline structured report JSON"
    )
    json_path: str | None = Field(
        default=None,
        description="Azure Blob Storage path to report JSON (container/blob)",
    )

    # Mode B: Raw document input
    file_path: str | None = Field(
        default=None,
        description="Azure Blob Storage path to PDF/DOCX for Document Intelligence extraction",
    )

    reference_report: str | None = Field(
        default=None, description="Optional gold-standard report"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "guideline_topic": "First-line treatment for transplant-eligible NDMM",
                    "disease_context": "Multiple Myeloma",
                    "report_json": {
                        "report_id": "rpt-001",
                        "sections": [
                            {
                                "id": "sec-1",
                                "title": "Introduction",
                                "content": "This guideline covers...",
                                "section_type": "general",
                                "order": 0,
                                "keywords": ["NDMM", "treatment"],
                            }
                        ],
                    },
                }
            ]
        }
    }
