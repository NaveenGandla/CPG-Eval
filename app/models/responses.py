from typing import Literal

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    score: int
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class SafetyMetricResult(MetricResult):
    missing_items: list[str] = []


class TraceabilityMetricResult(MetricResult):
    untraced_claims: list[dict] = Field(
        default=[], description="List of {claim, location} dicts"
    )


class FIHItem(BaseModel):
    claim: str
    source_says: str
    severity: Literal["critical", "major", "minor"]
    location: str


class EvaluationResponse(BaseModel):
    report_id: str
    evaluation_id: str
    timestamp: str
    evaluation_model: str
    num_runs: int
    metrics_evaluated: list[str]

    # Metric scores — None when the metric was not requested
    clinical_accuracy: MetricResult | None = None
    completeness: MetricResult | None = None
    safety_completeness: SafetyMetricResult | None = None
    relevance: MetricResult | None = None
    coherence: MetricResult | None = None
    evidence_traceability: TraceabilityMetricResult | None = None
    hallucination_score: MetricResult | None = None
    fih_detected: list[FIHItem] | None = None

    confidence_level: Literal["high", "medium", "low"]
    flags: list[str]

    # Storage references
    cosmos_document_id: str
    blob_url: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "report_id": "rpt-001",
                    "evaluation_id": "eval-abc-123",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "evaluation_model": "gpt-4o",
                    "num_runs": 1,
                    "metrics_evaluated": [
                        "clinical_accuracy",
                        "completeness",
                        "safety_completeness",
                        "relevance",
                        "coherence",
                        "evidence_traceability",
                        "hallucination_score",
                        "fih_detected",
                    ],
                    "clinical_accuracy": {
                        "score": 4,
                        "confidence": "high",
                        "reasoning": "All clinical facts verified against sources.",
                    },
                    "completeness": {
                        "score": 4,
                        "confidence": "medium",
                        "reasoning": "Covers main findings.",
                    },
                    "safety_completeness": {
                        "score": 3,
                        "confidence": "medium",
                        "reasoning": "Safety mentioned but lacks AE rates.",
                        "missing_items": ["Grade 3-4 AE rates"],
                    },
                    "relevance": {
                        "score": 5,
                        "confidence": "high",
                        "reasoning": "All content on-topic.",
                    },
                    "coherence": {
                        "score": 4,
                        "confidence": "high",
                        "reasoning": "Well-structured.",
                    },
                    "evidence_traceability": {
                        "score": 3,
                        "confidence": "medium",
                        "reasoning": "Some claims lack attribution.",
                        "untraced_claims": [],
                    },
                    "hallucination_score": {
                        "score": 3,
                        "confidence": "high",
                        "reasoning": "Minor statistical rounding.",
                    },
                    "fih_detected": [],
                    "confidence_level": "high",
                    "flags": ["missing_safety_data"],
                    "cosmos_document_id": "eval-abc-123",
                    "blob_url": "https://account.blob.core.windows.net/evaluation-reports/rpt-001/eval-abc-123.json",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Section-level evaluation response models
# ---------------------------------------------------------------------------


class SectionScore(BaseModel):
    """Evaluation scores for a single section."""

    section_id: str
    section_title: str
    section_type: str

    clinical_accuracy: MetricResult | None = None
    completeness: MetricResult | None = None
    safety_completeness: SafetyMetricResult | None = None
    relevance: MetricResult | None = None
    coherence: MetricResult | None = None
    evidence_traceability: TraceabilityMetricResult | None = None
    hallucination_score: MetricResult | None = None
    fih_detected: list[FIHItem] | None = None

    flags: list[str] = []


class SectionEvaluationResponse(BaseModel):
    """Response for section-wise evaluation."""

    report_id: str
    evaluation_id: str
    timestamp: str
    evaluation_model: str
    metrics_evaluated: list[str]

    # Aggregated final scores across all sections
    final_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated metric scores across all sections",
    )

    section_scores: list[SectionScore] = Field(
        default_factory=list,
        description="Per-section evaluation results",
    )

    confidence_level: Literal["high", "medium", "low"]
    flags: list[str]

    cosmos_document_id: str
    blob_url: str | None = None
