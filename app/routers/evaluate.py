"""POST /api/v1/evaluate endpoint and retrieval endpoints."""

import structlog
from fastapi import APIRouter, HTTPException

from app.models.requests import EvaluationRequest
from app.models.responses import EvaluationResponse
from app.services import cosmos_service
from app.services.evaluation_engine import run_evaluation

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluation"])


@router.post(
    "",
    response_model=EvaluationResponse,
    status_code=200,
    summary="Evaluate a CPG report",
    description=(
        "Run LLM-as-judge evaluation on a generated CPG report against source evidence. "
        "Supports 8 metrics: accuracy, hallucinations, consistency, source_traceability "
        "(percentage-based 0-100%), and coherence, clinical_relevance, bias, transparency "
        "(Likert 1-4)."
    ),
)
async def evaluate_report(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a generated CPG report against source evidence."""
    try:
        result = await run_evaluation(request)
        return result
    except RuntimeError as e:
        logger.error("evaluation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "throttl" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="Azure OpenAI service is throttled. Please retry later.",
            )
        logger.error("evaluation_error", error=error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {error_msg}",
        )


@router.get(
    "/{evaluation_id}",
    summary="Get evaluation by ID",
    description="Retrieve a stored evaluation from Cosmos DB by evaluation_id.",
)
async def get_evaluation(evaluation_id: str) -> dict:
    """Retrieve a stored evaluation by evaluation_id."""
    try:
        result = await cosmos_service.find_evaluation(evaluation_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation {evaluation_id} not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("cosmos_read_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation: {str(e)}",
        )


@router.get(
    "/report/{report_id}",
    summary="Get evaluations by report ID",
    description="Retrieve all evaluations for a given report_id.",
)
async def get_evaluations_by_report(report_id: str) -> list[dict]:
    """Retrieve all evaluations for a report_id."""
    try:
        results = await cosmos_service.get_evaluations_by_report(report_id)
        return results
    except Exception as e:
        logger.error("cosmos_read_error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluations: {str(e)}",
        )
