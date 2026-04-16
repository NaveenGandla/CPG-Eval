from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2024-10-21"

    # Azure Cosmos DB
    cosmos_endpoint: str = ""
    cosmos_database: str = "m42db"
    cosmos_container_evaluations: str = "evaluations"

    # Azure AI Search
    search_endpoint: str = ""
    search_index_name: str = "cpg-sources"

    # App Settings
    default_evaluation_model: str = "gpt-4o"
    log_level: str = "INFO"

    # Evaluation pipeline tuning
    claim_verification_batch_size: int = 5
    percentage_metric_top_k: int = 5
    likert_metric_top_k: int = 15
    max_concurrent_llm_calls: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
