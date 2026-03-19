"""Retrieve source chunks from Azure AI Search (optional enrichment)."""

import structlog
from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient

from app.config import settings
from app.models.requests import SourceChunk, SourceChunkMetadata

logger = structlog.get_logger()

_client: SearchClient | None = None


def get_search_client() -> SearchClient:
    """Get or create the Azure AI Search client."""
    global _client
    if _client is None:
        credential = DefaultAzureCredential()
        _client = SearchClient(
            endpoint=settings.search_endpoint,
            index_name=settings.search_index_name,
            credential=credential,
        )
    return _client


async def enrich_chunks(
    report_id: str, guideline_topic: str, top_k: int = 10
) -> list[SourceChunk]:
    """Query Azure AI Search to retrieve additional source chunks.

    This is a fallback when the caller doesn't provide retrieved_chunks.
    """
    if not settings.search_endpoint:
        logger.warning("search_not_configured")
        return []

    client = get_search_client()
    logger.info(
        "search_enrich",
        report_id=report_id,
        guideline_topic=guideline_topic,
        top_k=top_k,
    )

    chunks: list[SourceChunk] = []
    results = await client.search(
        search_text=guideline_topic,
        top=top_k,
    )

    async for result in results:
        chunk = SourceChunk(
            chunk_id=result.get("id", ""),
            text=result.get("content", result.get("text", "")),
            metadata=SourceChunkMetadata(
                study_name=result.get("study_name"),
                year=result.get("year"),
                journal=result.get("journal"),
                authors=result.get("authors"),
            ),
        )
        chunks.append(chunk)

    logger.info("search_enrich_done", report_id=report_id, num_chunks=len(chunks))
    return chunks
