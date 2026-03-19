"""Tests for the /api/v1/evaluate endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.responses import (
    EvaluationResponse,
    FIHItem,
    MetricResult,
    SafetyMetricResult,
    TraceabilityMetricResult,
)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "report_id": "rpt-001",
        "generated_report": "This report discusses treatment options for MM.",
        "retrieved_chunks": [
            {
                "chunk_id": "c1",
                "text": "The GRIFFIN trial showed D-VRd is effective.",
                "metadata": {"study_name": "GRIFFIN", "year": 2023},
            }
        ],
        "guideline_topic": "First-line treatment for NDMM",
        "disease_context": "Multiple Myeloma",
        "num_eval_runs": 1,
    }


@pytest.fixture
def mock_evaluation_response() -> EvaluationResponse:
    return EvaluationResponse(
        report_id="rpt-001",
        evaluation_id="eval-123",
        timestamp="2025-01-15T10:30:00Z",
        model_used="gpt-4o",
        num_runs=1,
        clinical_accuracy=MetricResult(
            score=4, confidence="high", reasoning="Good"
        ),
        completeness=MetricResult(
            score=3, confidence="medium", reasoning="OK"
        ),
        safety_completeness=SafetyMetricResult(
            score=3, confidence="medium", reasoning="OK"
        ),
        relevance=MetricResult(
            score=5, confidence="high", reasoning="Great"
        ),
        coherence=MetricResult(
            score=4, confidence="high", reasoning="Good"
        ),
        evidence_traceability=TraceabilityMetricResult(
            score=3, confidence="medium", reasoning="OK"
        ),
        hallucination_score=MetricResult(
            score=3, confidence="high", reasoning="Clean"
        ),
        fih_detected=[],
        overall_score=72.5,
        usable_without_editing=False,
        confidence_level="medium",
        flags=[],
        cosmos_document_id="eval-123",
        blob_url=None,
    )


@pytest.mark.asyncio
class TestEvaluateEndpoint:
    @patch("app.routers.evaluate.run_evaluation")
    async def test_valid_request(
        self, mock_run, valid_payload, mock_evaluation_response
    ):
        mock_run.return_value = mock_evaluation_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=valid_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["report_id"] == "rpt-001"
        assert "overall_score" in data
        assert "clinical_accuracy" in data

    async def test_missing_report(self, valid_payload):
        payload = valid_payload.copy()
        del payload["generated_report"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422  # Validation error

    async def test_empty_report(self, valid_payload):
        payload = valid_payload.copy()
        payload["generated_report"] = ""

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    async def test_empty_chunks(self, valid_payload):
        payload = valid_payload.copy()
        payload["retrieved_chunks"] = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    async def test_num_eval_runs_out_of_range(self, valid_payload):
        payload = valid_payload.copy()
        payload["num_eval_runs"] = 10  # max is 7

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    @patch("app.routers.evaluate.run_evaluation")
    async def test_server_error(self, mock_run, valid_payload):
        mock_run.side_effect = RuntimeError("All runs failed")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=valid_payload)

        assert response.status_code == 500

    @patch("app.routers.evaluate.run_evaluation")
    async def test_throttling_error(self, mock_run, valid_payload):
        mock_run.side_effect = Exception("429 Too Many Requests")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=valid_payload)

        assert response.status_code == 503


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "version": "1.0.0"}


@pytest.mark.asyncio
class TestGetEndpoints:
    @patch("app.routers.evaluate.cosmos_service")
    async def test_get_evaluation_not_found(self, mock_cosmos):
        mock_cosmos.find_evaluation = AsyncMock(return_value=None)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/evaluate/nonexistent-id")

        assert response.status_code == 404

    @patch("app.routers.evaluate.cosmos_service")
    async def test_get_evaluations_by_report(self, mock_cosmos):
        mock_cosmos.get_evaluations_by_report = AsyncMock(return_value=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/evaluate/report/rpt-001")

        assert response.status_code == 200
        assert response.json() == []
