"""Store and retrieve evaluation results in Azure Cosmos DB."""

import structlog
from azure.cosmos.aio import ContainerProxy, CosmosClient
from azure.identity import DefaultAzureCredential

from app.config import settings

logger = structlog.get_logger()

_client: CosmosClient | None = None
_container: ContainerProxy | None = None


async def get_container() -> ContainerProxy:
    """Get or create the Cosmos DB container client."""
    global _client, _container
    if _container is None:
        credential = DefaultAzureCredential()
        _client = CosmosClient(settings.cosmos_endpoint, credential=credential)
        database = _client.get_database_client(settings.cosmos_database)
        _container = database.get_container_client(
            settings.cosmos_container_evaluations
        )
    return _container


async def store_evaluation(document: dict) -> str:
    """Store an evaluation document in Cosmos DB.

    The document must include 'id' and 'report_id' (partition key).
    Returns the document id.
    """
    container = await get_container()
    logger.info(
        "cosmos_store",
        evaluation_id=document.get("id"),
        report_id=document.get("report_id"),
    )
    await container.create_item(body=document)
    return document["id"]


async def get_evaluation(evaluation_id: str, report_id: str) -> dict | None:
    """Retrieve a single evaluation by evaluation_id.

    Requires report_id as partition key for efficient lookup.
    """
    container = await get_container()
    try:
        item = await container.read_item(
            item=evaluation_id, partition_key=report_id
        )
        return item
    except Exception:
        logger.warning(
            "cosmos_not_found",
            evaluation_id=evaluation_id,
            report_id=report_id,
        )
        return None


async def get_evaluations_by_report(report_id: str) -> list[dict]:
    """Retrieve all evaluations for a given report_id."""
    container = await get_container()
    query = "SELECT * FROM c WHERE c.report_id = @report_id"
    parameters = [{"name": "@report_id", "value": report_id}]
    items: list[dict] = []
    async for item in container.query_items(
        query=query,
        parameters=parameters,
        partition_key=report_id,
    ):
        items.append(item)
    return items


async def find_evaluation(evaluation_id: str) -> dict | None:
    """Find an evaluation by evaluation_id across all partitions (cross-partition query)."""
    container = await get_container()
    query = "SELECT * FROM c WHERE c.evaluation_id = @eval_id"
    parameters = [{"name": "@eval_id", "value": evaluation_id}]
    async for item in container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
    ):
        return item
    return None
