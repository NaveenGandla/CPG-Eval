"""Store full evaluation reports in Azure Blob Storage."""

import json

import structlog
from azure.identity import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

from app.config import settings

logger = structlog.get_logger()

_client: BlobServiceClient | None = None


def get_blob_client() -> BlobServiceClient:
    """Get or create the Blob Storage service client."""
    global _client
    if _client is None:
        credential = DefaultAzureCredential()
        _client = BlobServiceClient(
            account_url=settings.blob_account_url, credential=credential
        )
    return _client


async def store_evaluation_report(
    report_id: str, evaluation_id: str, data: dict
) -> str | None:
    """Store a full evaluation JSON to blob storage.

    Path: evaluation-reports/{report_id}/{evaluation_id}.json
    Returns the blob URL or None if storage is not configured.
    """
    if not settings.blob_account_url:
        logger.warning("blob_not_configured")
        return None

    blob_path = f"{report_id}/{evaluation_id}.json"
    client = get_blob_client()
    container_client = client.get_container_client(settings.blob_container_name)
    blob_client = container_client.get_blob_client(blob_path)

    content = json.dumps(data, indent=2, default=str)
    await blob_client.upload_blob(content, overwrite=True)

    blob_url = f"{settings.blob_account_url}/{settings.blob_container_name}/{blob_path}"
    logger.info(
        "blob_stored",
        report_id=report_id,
        evaluation_id=evaluation_id,
        blob_url=blob_url,
    )
    return blob_url
