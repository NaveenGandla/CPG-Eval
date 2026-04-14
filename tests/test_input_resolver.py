"""Tests for input resolver."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.requests import (
    ReportJSON,
    ReportSection,
    SectionEvaluationRequest,
)
from app.services.input_resolver import resolve_to_json


@pytest.fixture
def inline_report_json() -> ReportJSON:
    return ReportJSON(
        report_id="rpt-inline-001",
        sections=[
            ReportSection(
                id="sec-1",
                title="Test Section",
                content="Some test content here.",
                section_type="general",
                order=0,
                keywords=["test"],
            ),
        ],
    )


@pytest.fixture
def inline_report_no_keywords() -> ReportJSON:
    return ReportJSON(
        report_id="rpt-no-kw-001",
        sections=[
            ReportSection(
                id="sec-1",
                title="Treatment Recommendations",
                content=(
                    "D-VRd is recommended as first-line therapy for transplant-eligible patients. "
                    "The GRIFFIN trial demonstrated an overall response rate of 99%. "
                    "Bortezomib-based regimens remain the standard of care for newly diagnosed "
                    "multiple myeloma patients who are eligible for autologous stem cell transplant."
                ),
                section_type="guideline",
                order=0,
            ),
        ],
    )


@pytest.mark.asyncio
class TestResolveToJson:
    async def test_inline_json(self, inline_report_json):
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            report_json=inline_report_json,
        )
        result = await resolve_to_json(request)
        assert result.report_id == "rpt-inline-001"
        assert len(result.sections) == 1

    async def test_no_input_raises_error(self):
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
        )
        with pytest.raises(ValueError, match="No input provided"):
            await resolve_to_json(request)

    @patch("app.services.input_resolver.settings")
    async def test_blob_json_path_requires_config(self, mock_settings):
        mock_settings.blob_account_url = ""
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            json_path="container/report.json",
        )
        with pytest.raises(RuntimeError, match="Blob Storage not configured"):
            await resolve_to_json(request)

    @patch("app.services.input_resolver.settings")
    async def test_file_path_requires_di_config(self, mock_settings):
        mock_settings.document_intelligence_endpoint = ""
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            file_path="https://blob.example.com/docs/report.pdf",
        )
        with pytest.raises(RuntimeError, match="Document Intelligence endpoint not configured"):
            await resolve_to_json(request)

    async def test_inline_json_priority_over_paths(self, inline_report_json):
        """When report_json is provided, it takes priority over json_path and file_path."""
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            report_json=inline_report_json,
            json_path="some/path.json",
            file_path="https://blob.example.com/doc.pdf",
        )
        result = await resolve_to_json(request)
        assert result.report_id == "rpt-inline-001"

    async def test_auto_generates_keywords_when_missing(self, inline_report_no_keywords):
        """Keywords are auto-generated via TF-IDF when not provided in JSON."""
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            report_json=inline_report_no_keywords,
        )
        result = await resolve_to_json(request)
        section = result.sections[0]
        assert len(section.keywords) > 0, "Keywords should be auto-generated"

    async def test_preserves_user_provided_keywords(self, inline_report_json):
        """User-provided keywords are not overwritten."""
        request = SectionEvaluationRequest(
            guideline_topic="Treatment",
            disease_context="MM",
            report_json=inline_report_json,
        )
        result = await resolve_to_json(request)
        assert result.sections[0].keywords == ["test"]
