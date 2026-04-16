"""Tests for the /api/v1/evaluate endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.responses import (
    EvaluationResponse,
    LikertMetricResult,
    LikertSubQuestionScore,
    PercentageMetricResult,
)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "report_id": "rpt-001",
        "generated_report": "This report discusses treatment options for MM.",
        "guideline_topic": "First-line treatment for NDMM",
        "disease_context": "Multiple Myeloma",
    }


@pytest.fixture
def mock_evaluation_response() -> EvaluationResponse:
    return EvaluationResponse(
        report_id="rpt-001",
        evaluation_id="eval-123",
        timestamp="2025-01-15T10:30:00Z",
        evaluation_model="gpt-4o",
        metrics_evaluated=[
            "accuracy",
            "hallucinations",
            "consistency",
            "source_traceability",
            "coherence",
            "clinical_relevance",
            "bias",
            "transparency",
        ],
        accuracy=PercentageMetricResult(score=85.0),
        hallucinations=PercentageMetricResult(score=90.0),
        consistency=PercentageMetricResult(score=95.0),
        source_traceability=PercentageMetricResult(score=80.0),
        coherence=LikertMetricResult(
            score=3.5,
            sub_questions=[
                LikertSubQuestionScore(
                    sub_question_id="coherence_pathway_alignment",
                    sub_question_text="Pathway aligns.",
                    score=4,
                    reasoning="Good.",
                )
            ],
            overall_reasoning="Well structured.",
        ),
        clinical_relevance=LikertMetricResult(score=3.0, overall_reasoning="OK"),
        bias=LikertMetricResult(score=3.0, overall_reasoning="Fair"),
        transparency=LikertMetricResult(score=3.25, overall_reasoning="Clear"),
        flags=[],
        cosmos_document_id="eval-123",
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
        assert "accuracy" in data
        assert "coherence" in data
        assert "metrics_evaluated" in data

    @patch("app.routers.evaluate.run_evaluation")
    async def test_valid_request_with_selected_metrics(
        self, mock_run, valid_payload, mock_evaluation_response
    ):
        """User can select a subset of metrics."""
        payload = valid_payload.copy()
        payload["metrics"] = ["accuracy", "coherence"]
        mock_run.return_value = mock_evaluation_response

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 200

    async def test_missing_report(self, valid_payload):
        payload = valid_payload.copy()
        del payload["generated_report"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    async def test_empty_report(self, valid_payload):
        payload = valid_payload.copy()
        payload["generated_report"] = ""

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    async def test_empty_metrics_list(self, valid_payload):
        payload = valid_payload.copy()
        payload["metrics"] = []

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/evaluate", json=payload)

        assert response.status_code == 422

    async def test_invalid_metric_name(self, valid_payload):
        payload = valid_payload.copy()
        payload["metrics"] = ["accuracy", "nonexistent_metric"]

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

    async def test_sections_endpoint_removed(self):
        """The /sections endpoint should no longer exist."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/evaluate/sections",
                json={"guideline_topic": "x", "disease_context": "y"},
            )

        # Should be 405 (method not allowed on the GET /{eval_id} route) or 422
        assert response.status_code in (404, 405, 422)
