from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Percentage metric models (claim-level verification)
# ---------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    claim_id: str
    claim_text: str
    location: str


class ClaimVerdict(BaseModel):
    claim_id: str
    verdict: Literal["correct", "incorrect", "unverifiable"]
    reasoning: str
    evidence_chunk_id: str | None = None
    conflicting_location: str | None = None


class SubQuestionResult(BaseModel):
    sub_question_id: str
    sub_question_text: str
    claims_extracted: list[ExtractedClaim] = []
    verifications: list[ClaimVerdict] = []
    correct_count: int = 0
    total_count: int = 0
    percentage: float = Field(
        default=100.0,
        description="Score as percentage 0-100. Defaults to 100 when no claims found.",
    )


class PercentageMetricResult(BaseModel):
    score: float = Field(..., description="Aggregated score 0-100%")
    sub_questions: list[SubQuestionResult] = []


# ---------------------------------------------------------------------------
# Likert metric models (1-4 scale)
# ---------------------------------------------------------------------------


class LikertSubQuestionScore(BaseModel):
    sub_question_id: str
    sub_question_text: str
    score: int = Field(..., ge=1, le=4)
    reasoning: str


class LikertMetricResult(BaseModel):
    score: float = Field(..., description="Average score 1.0-4.0")
    sub_questions: list[LikertSubQuestionScore] = []
    overall_reasoning: str = ""


# ---------------------------------------------------------------------------
# Top-level evaluation response
# ---------------------------------------------------------------------------


class EvaluationResponse(BaseModel):
    report_id: str
    evaluation_id: str
    timestamp: str
    evaluation_model: str
    metrics_evaluated: list[str]

    # Percentage metrics (0-100%)
    accuracy: PercentageMetricResult | None = None
    hallucinations: PercentageMetricResult | None = None
    consistency: PercentageMetricResult | None = None
    source_traceability: PercentageMetricResult | None = None

    # Likert metrics (1-4)
    coherence: LikertMetricResult | None = None
    clinical_relevance: LikertMetricResult | None = None
    bias: LikertMetricResult | None = None
    transparency: LikertMetricResult | None = None

    flags: list[str] = []

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
                    "metrics_evaluated": [
                        "accuracy",
                        "hallucinations",
                        "coherence",
                        "transparency",
                    ],
                    "accuracy": {
                        "score": 85.0,
                        "sub_questions": [
                            {
                                "sub_question_id": "accuracy_drug_dosages",
                                "sub_question_text": "Are drug dosages, frequencies, and routes of administration correct?",
                                "claims_extracted": [
                                    {
                                        "claim_id": "c1",
                                        "claim_text": "Lenalidomide 25 mg on days 1-21",
                                        "location": "Section 3, paragraph 2",
                                    }
                                ],
                                "verifications": [
                                    {
                                        "claim_id": "c1",
                                        "verdict": "correct",
                                        "reasoning": "Matches source evidence from GRIFFIN trial.",
                                        "evidence_chunk_id": "chunk-5",
                                    }
                                ],
                                "correct_count": 1,
                                "total_count": 1,
                                "percentage": 100.0,
                            }
                        ],
                    },
                    "coherence": {
                        "score": 3.25,
                        "sub_questions": [
                            {
                                "sub_question_id": "coherence_pathway_alignment",
                                "sub_question_text": "The clinical pathway aligns with the recommendations in the guideline.",
                                "score": 3,
                                "reasoning": "Pathway mostly aligns but minor gaps in staging criteria.",
                            }
                        ],
                        "overall_reasoning": "Document is well structured with minor gaps.",
                    },
                    "flags": ["low_accuracy"],
                    "cosmos_document_id": "eval-abc-123",
                    "blob_url": "https://account.blob.core.windows.net/evaluation-reports/rpt-001/eval-abc-123.json",
                }
            ]
        }
    }
