"""Resolve various input formats into a normalized ReportJSON structure."""

import json

import structlog
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

from app.config import settings
from app.models.requests import ReportJSON, ReportSection, SectionEvaluationRequest
from app.services.document_intelligence import extract_from_blob
from app.services.section_builder import build_sections
from app.utils.keyword_extraction import extract_keywords

logger = structlog.get_logger()


async def resolve_to_json(request: SectionEvaluationRequest) -> ReportJSON:
    """Resolve input data to a normalized ReportJSON.

    Supports three modes:
    - report_json: return directly (already structured)
    - json_path: load from Azure Blob Storage
    - file_path: extract via Document Intelligence, then build sections
    """
    if request.report_json is not None:
        logger.info(
            "input_resolve_inline_json",
            report_id=request.report_json.report_id,
        )
        _enrich_missing_keywords(request.report_json)
        return request.report_json

    if request.json_path is not None:
        logger.info("input_resolve_blob_json", json_path=request.json_path)
        report = await _load_json_from_blob(request.json_path)
        _enrich_missing_keywords(report)
        return report

    if request.file_path is not None:
        logger.info("input_resolve_document", file_path=request.file_path)
        return await _extract_and_build(request.file_path)

    raise ValueError(
        "No input provided. Supply one of: report_json, json_path, or file_path."
    )


def _enrich_missing_keywords(report: ReportJSON) -> None:
    """Auto-generate TF-IDF keywords for sections that don't have them.

    Mutates sections in-place so the same extraction used for PDF inputs
    is applied consistently to JSON inputs.
    """
    enriched = 0
    for section in report.sections:
        if not section.keywords:
            section.keywords = extract_keywords(section.content, top_n=10)
            enriched += 1

    if enriched:
        logger.info(
            "keywords_enriched",
            report_id=report.report_id,
            sections_enriched=enriched,
        )


async def _load_json_from_blob(blob_path: str) -> ReportJSON:
    """Load a ReportJSON from Azure Blob Storage.

    Args:
        blob_path: Path in the format 'container/blob_name' or just 'blob_name'
                   (uses configured container).
    """
    if not settings.blob_account_url:
        raise RuntimeError(
            "Blob Storage not configured. Set BLOB_ACCOUNT_URL environment variable."
        )

    # Parse container/blob from path
    parts = blob_path.split("/", 1)
    if len(parts) == 2:
        container_name, blob_name = parts
    else:
        container_name = settings.blob_json_container_name
        blob_name = parts[0]

    credential = DefaultAzureCredential()
    async with BlobServiceClient(
        account_url=settings.blob_account_url, credential=credential
    ) as client:
        container_client = client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        download = await blob_client.download_blob()
        raw = await download.readall()

    data = json.loads(raw)
    report = ReportJSON(**data)

    logger.info(
        "blob_json_loaded",
        report_id=report.report_id,
        num_sections=len(report.sections),
    )
    return report


async def _extract_and_build(file_path: str) -> ReportJSON:
    """Extract document via Document Intelligence and build sections.

    Args:
        file_path: Azure Blob Storage URL to the PDF/DOCX file.
    """
    # Step 1: Extract layout via Document Intelligence
    layout_output = await extract_from_blob(file_path)

    # Step 2: Build sections from the extracted layout
    sections: list[ReportSection] = build_sections(layout_output)

    if not sections:
        raise RuntimeError(
            f"No sections could be extracted from document: {file_path}"
        )

    # Generate a report_id from the file path
    import hashlib

    report_id = f"doc-{hashlib.sha256(file_path.encode()).hexdigest()[:12]}"

    report = ReportJSON(report_id=report_id, sections=sections)

    logger.info(
        "document_sections_built",
        report_id=report_id,
        file_path=file_path,
        num_sections=len(sections),
    )

    # Bonus: save generated JSON back to blob for reuse
    await _save_json_to_blob(report)

    return report


async def _save_json_to_blob(report: ReportJSON) -> str | None:
    """Save generated ReportJSON to blob storage for reuse."""
    if not settings.blob_account_url:
        return None

    try:
        blob_path = f"{report.report_id}.json"
        credential = DefaultAzureCredential()
        async with BlobServiceClient(
            account_url=settings.blob_account_url, credential=credential
        ) as client:
            container_client = client.get_container_client(
                settings.blob_json_container_name
            )
            blob_client = container_client.get_blob_client(blob_path)

            content = report.model_dump_json(indent=2)
            await blob_client.upload_blob(content, overwrite=True)

        url = f"{settings.blob_account_url}/{settings.blob_json_container_name}/{blob_path}"
        logger.info("json_saved_to_blob", report_id=report.report_id, blob_url=url)
        return url
    except Exception as e:
        logger.warning("json_blob_save_failed", error=str(e))
        return None
