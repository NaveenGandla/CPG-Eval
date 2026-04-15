from typing import Literal

from pydantic import BaseModel, Field

# All available metric names for multi-select
MetricName = Literal[
    "accuracy",
    "hallucinations",
    "consistency",
    "source_traceability",
    "coherence",
    "clinical_relevance",
    "bias",
    "transparency",
]

ALL_METRICS: list[str] = [
    "accuracy",
    "hallucinations",
    "consistency",
    "source_traceability",
    "coherence",
    "clinical_relevance",
    "bias",
    "transparency",
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
            "Metrics to evaluate. Choose from: accuracy, hallucinations, "
            "consistency, source_traceability, coherence, clinical_relevance, "
            "bias, transparency. Defaults to all."
        ),
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
                        "accuracy",
                        "hallucinations",
                        "coherence",
                        "transparency",
                    ],
                }
            ]
        }
    }
