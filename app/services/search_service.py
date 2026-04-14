"""Retrieve source chunks from Azure AI Search."""

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
    report_id: str, guideline_topic: str, disease_context: str, top_k: int = 10
) -> list[SourceChunk]:
    """Query Azure AI Search to retrieve source evidence chunks.

    Combines guideline_topic and disease_context for a more targeted search.
    """
    if not settings.search_endpoint:
        logger.warning("search_not_configured")
        return []

    client = get_search_client()
    search_query = f"{disease_context} {guideline_topic}"
    logger.info(
        "search_enrich",
        report_id=report_id,
        search_query=search_query,
        top_k=top_k,
    )

    chunks: list[SourceChunk] = []
    results = await client.search(
        search_text=search_query,
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


async def retrieve_for_section(
    section: dict,
    top_k: int = 5,
    disease_context: str = "",
    guideline_topic: str = "",
) -> list[SourceChunk]:
    """Retrieve source chunks specific to a single report section.

    Builds a targeted query from the section title, keywords, and
    the broader disease/guideline context for improved relevance.
    """
    if not settings.search_endpoint:
        logger.warning("search_not_configured")
        return []

    title = section.get("title", "")
    keywords = section.get("keywords", [])

    # Build a richer query: context terms first (boost relevance),
    # then section title and keywords
    query_parts: list[str] = []
    if disease_context:
        query_parts.append(disease_context)
    if guideline_topic:
        query_parts.append(guideline_topic)
    if title:
        query_parts.append(title)
    if keywords:
        query_parts.append(" ".join(keywords))

    search_query = " ".join(query_parts)

    client = get_search_client()
    logger.info(
        "search_section_retrieve",
        section_id=section.get("id", ""),
        search_query=search_query,
        top_k=top_k,
    )

    chunks: list[SourceChunk] = []
    results = await client.search(
        search_text=search_query,
        top=top_k,
        search_mode="all",
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

    logger.info(
        "search_section_retrieve_done",
        section_id=section.get("id", ""),
        num_chunks=len(chunks),
    )
    return chunks
