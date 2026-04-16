"""Shared test fixtures."""

import pytest

from app.models.requests import (
    EvaluationRequest,
    SourceChunk,
    SourceChunkMetadata,
)
from app.models.responses import (
    ClaimVerdict,
    ExtractedClaim,
    LikertMetricResult,
    LikertSubQuestionScore,
    PercentageMetricResult,
    SubQuestionResult,
)


@pytest.fixture
def sample_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="chunk-1",
            text="The GRIFFIN trial demonstrated that D-VRd followed by ASCT improved PFS significantly.",
            metadata=SourceChunkMetadata(
                study_name="GRIFFIN", year=2023, journal="Blood"
            ),
        ),
        SourceChunk(
            chunk_id="chunk-2",
            text="Daratumumab plus VRd showed an ORR of 99% in transplant-eligible NDMM patients.",
            metadata=SourceChunkMetadata(
                study_name="GRIFFIN", year=2023, journal="Blood"
            ),
        ),
    ]


@pytest.fixture
def sample_request() -> EvaluationRequest:
    return EvaluationRequest(
        report_id="rpt-test-001",
        generated_report=(
            "This CPG report covers first-line treatment options for newly "
            "diagnosed multiple myeloma. The GRIFFIN trial demonstrated that "
            "D-VRd followed by ASCT improved PFS. The ORR was 99%."
        ),
        guideline_topic="First-line treatment for transplant-eligible NDMM",
        disease_context="Multiple Myeloma",
    )


@pytest.fixture
def sample_percentage_result() -> PercentageMetricResult:
    return PercentageMetricResult(
        score=75.0,
        sub_questions=[
            SubQuestionResult(
                sub_question_id="accuracy_drug_dosages",
                sub_question_text="Are drug dosages correct?",
                claims_extracted=[
                    ExtractedClaim(
                        claim_id="c1",
                        claim_text="Lenalidomide 25 mg daily",
                        location="Section 3",
                    )
                ],
                verifications=[
                    ClaimVerdict(
                        claim_id="c1",
                        verdict="correct",
                        reasoning="Matches source.",
                        evidence_chunk_id="chunk-1",
                    )
                ],
                correct_count=1,
                total_count=1,
                percentage=100.0,
            ),
            SubQuestionResult(
                sub_question_id="accuracy_lab_ranges",
                sub_question_text="Are lab ranges correct?",
                claims_extracted=[
                    ExtractedClaim(claim_id="c1", claim_text="eGFR > 30", location="Section 2"),
                    ExtractedClaim(claim_id="c2", claim_text="Platelets > 100k", location="Section 2"),
                ],
                verifications=[
                    ClaimVerdict(claim_id="c1", verdict="correct", reasoning="OK"),
                    ClaimVerdict(claim_id="c2", verdict="incorrect", reasoning="Should be >75k"),
                ],
                correct_count=1,
                total_count=2,
                percentage=50.0,
            ),
        ],
    )


@pytest.fixture
def sample_likert_result() -> LikertMetricResult:
    return LikertMetricResult(
        score=3.25,
        sub_questions=[
            LikertSubQuestionScore(
                sub_question_id="coherence_pathway_alignment",
                sub_question_text="Pathway aligns with recommendations.",
                score=3,
                reasoning="Mostly aligned.",
            ),
            LikertSubQuestionScore(
                sub_question_id="coherence_sections",
                sub_question_text="Sections support each other.",
                score=4,
                reasoning="Strong coherence.",
            ),
            LikertSubQuestionScore(
                sub_question_id="coherence_terminology",
                sub_question_text="Terminology is consistent.",
                score=3,
                reasoning="Minor inconsistencies.",
            ),
            LikertSubQuestionScore(
                sub_question_id="coherence_unified",
                sub_question_text="Reads as unified guideline.",
                score=3,
                reasoning="Mostly unified.",
            ),
        ],
        overall_reasoning="Document is well structured with minor gaps.",
    )
