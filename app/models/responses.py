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
    model_used: str
    num_runs: int

    # Metric scores (aggregated via majority voting / median)
    clinical_accuracy: MetricResult
    completeness: MetricResult
    safety_completeness: SafetyMetricResult
    relevance: MetricResult
    coherence: MetricResult
    evidence_traceability: TraceabilityMetricResult
    hallucination_score: MetricResult
    fih_detected: list[FIHItem]

    # Aggregate
    overall_score: float = Field(
        ..., ge=0, le=100, description="0-100 weighted score"
    )
    usable_without_editing: bool
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
                    "model_used": "gpt-4o",
                    "num_runs": 3,
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
                    "overall_score": 78.5,
                    "usable_without_editing": False,
                    "confidence_level": "medium",
                    "flags": ["missing_safety_data"],
                    "cosmos_document_id": "eval-abc-123",
                    "blob_url": "https://account.blob.core.windows.net/evaluation-reports/rpt-001/eval-abc-123.json",
                }
            ]
        }
    }
