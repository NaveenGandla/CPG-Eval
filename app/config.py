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

    # Azure Blob Storage
    blob_account_url: str = ""
    blob_container_name: str = "evaluation-reports"

    # App Settings
    default_num_eval_runs: int = 3
    default_evaluation_model: str = "gpt-4o"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
