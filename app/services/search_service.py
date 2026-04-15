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


async def _search_and_parse(search_query: str, top_k: int) -> list[SourceChunk]:
    """Execute a search query and parse results into SourceChunk objects."""
    client = get_search_client()
    chunks: list[SourceChunk] = []
    results = await client.search(search_text=search_query, top=top_k)

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

    return chunks


async def enrich_chunks(
    report_id: str, guideline_topic: str, disease_context: str, top_k: int = 15
) -> list[SourceChunk]:
    """Retrieve document-level source evidence chunks for Likert evaluation."""
    if not settings.search_endpoint:
        logger.warning("search_not_configured")
        return []

    search_query = f"{disease_context} {guideline_topic}"
    logger.info(
        "search_enrich",
        report_id=report_id,
        search_query=search_query,
        top_k=top_k,
    )

    chunks = await _search_and_parse(search_query, top_k)

    logger.info("search_enrich_done", report_id=report_id, num_chunks=len(chunks))
    return chunks


async def retrieve_for_claim(
    claim_text: str,
    disease_context: str,
    guideline_topic: str,
    top_k: int = 5,
) -> list[SourceChunk]:
    """Retrieve source chunks specific to a single claim for verification."""
    if not settings.search_endpoint:
        logger.warning("search_not_configured")
        return []

    # Build query: claim text augmented with context for relevance
    query_parts: list[str] = []
    if disease_context:
        query_parts.append(disease_context)
    if guideline_topic:
        query_parts.append(guideline_topic)
    query_parts.append(claim_text)
    search_query = " ".join(query_parts)

    logger.debug(
        "search_claim_retrieve",
        claim_text=claim_text[:100],
        top_k=top_k,
    )

    chunks = await _search_and_parse(search_query, top_k)

    logger.debug(
        "search_claim_retrieve_done",
        claim_text=claim_text[:50],
        num_chunks=len(chunks),
    )
    return chunks
