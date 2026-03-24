"""Shared test fixtures."""

import pytest

from app.models.requests import (
    EvaluationRequest,
    SourceChunk,
    SourceChunkMetadata,
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
        model="gpt-4o",
    )


@pytest.fixture
def sample_llm_judge_result() -> dict:
    """A single LLM judge evaluation result (all 8 metrics)."""
    return {
        "clinical_accuracy": {
            "score": 4,
            "confidence": "high",
            "reasoning": "Clinical facts verified.",
        },
        "completeness": {
            "score": 3,
            "confidence": "medium",
            "reasoning": "Covers main findings but misses comparator data.",
        },
        "safety_completeness": {
            "score": 2,
            "confidence": "medium",
            "reasoning": "Safety data is superficial.",
            "missing_items": ["Grade 3-4 AE rates", "Black box warnings"],
        },
        "relevance": {
            "score": 5,
            "confidence": "high",
            "reasoning": "All content on-topic.",
        },
        "coherence": {
            "score": 4,
            "confidence": "high",
            "reasoning": "Well-structured document.",
        },
        "evidence_traceability": {
            "score": 3,
            "confidence": "medium",
            "reasoning": "Some claims lack attribution.",
            "untraced_claims": [
                {"claim": "ORR was 99%", "location": "paragraph 1"}
            ],
        },
        "hallucination_score": {
            "score": 3,
            "confidence": "high",
            "reasoning": "No major hallucinations detected.",
        },
        "fih_detected": [],
    }
