"""Extract structured text from PDF/DOCX using Azure Document Intelligence."""

import structlog
from azure.identity import DefaultAzureCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

from app.config import settings

logger = structlog.get_logger()

_client: DocumentIntelligenceClient | None = None


def get_di_client() -> DocumentIntelligenceClient:
    """Get or create the Document Intelligence client."""
    global _client
    if _client is None:
        credential = DefaultAzureCredential()
        _client = DocumentIntelligenceClient(
            endpoint=settings.document_intelligence_endpoint,
            credential=credential,
        )
    return _client


async def extract_from_blob(blob_url: str) -> dict:
    """Extract layout from a document in blob storage using prebuilt-layout.

    Args:
        blob_url: Full URL to the document in Azure Blob Storage.

    Returns:
        Dict with 'paragraphs' and 'tables' extracted from the document.
    """
    if not settings.document_intelligence_endpoint:
        raise RuntimeError(
            "Document Intelligence endpoint not configured. "
            "Set DOCUMENT_INTELLIGENCE_ENDPOINT environment variable."
        )

    client = get_di_client()

    logger.info("di_extract_start", blob_url=blob_url)

    # The SDK's begin_analyze_document is synchronous — run in thread for async compat
    import asyncio

    loop = asyncio.get_event_loop()
    poller = await loop.run_in_executor(
        None,
        lambda: client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(url_source=blob_url),
        ),
    )
    result = await loop.run_in_executor(None, poller.result)

    paragraphs = []
    tables = []

    # Extract paragraphs with role metadata
    if result.paragraphs:
        for para in result.paragraphs:
            paragraphs.append(
                {
                    "content": para.content,
                    "role": getattr(para, "role", None),  # title, sectionHeading, etc.
                }
            )

    # Extract tables and flatten to text
    if result.tables:
        for table in result.tables:
            rows: list[list[str]] = []
            if table.cells:
                max_row = max(c.row_index for c in table.cells) + 1
                max_col = max(c.column_index for c in table.cells) + 1
                rows = [[""] * max_col for _ in range(max_row)]
                for cell in table.cells:
                    rows[cell.row_index][cell.column_index] = cell.content

            # Flatten table to text representation
            table_text_lines = []
            for row in rows:
                table_text_lines.append(" | ".join(row))
            tables.append("\n".join(table_text_lines))

    logger.info(
        "di_extract_done",
        num_paragraphs=len(paragraphs),
        num_tables=len(tables),
    )

    return {"paragraphs": paragraphs, "tables": tables}
