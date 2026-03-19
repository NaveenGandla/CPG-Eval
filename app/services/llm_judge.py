"""Azure OpenAI LLM judge calls with structured prompts."""

import json

import structlog
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

from app.config import settings
from app.prompts.evaluation_prompts import SYSTEM_PROMPT

logger = structlog.get_logger()

_client: AsyncAzureOpenAI | None = None


def get_openai_client() -> AsyncAzureOpenAI:
    """Get or create the Azure OpenAI async client."""
    global _client
    if _client is None:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        _client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.azure_openai_api_version,
        )
    return _client


async def call_llm_judge(
    user_prompt: str,
    deployment: str,
    report_id: str,
    run_index: int,
    max_retries: int = 2,
) -> dict:
    """Call Azure OpenAI with the evaluation prompt and return parsed JSON.

    Retries up to max_retries times if the response is not valid JSON.
    """
    client = get_openai_client()
    last_error: Exception | None = None

    for attempt in range(1 + max_retries):
        prompt = user_prompt
        if attempt > 0:
            prompt += (
                "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "You MUST respond with ONLY valid JSON, no markdown fences, "
                "no preamble, no explanation outside the JSON object."
            )

        try:
            logger.info(
                "llm_judge_call",
                report_id=report_id,
                run_index=run_index,
                attempt=attempt,
                deployment=deployment,
            )
            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
                timeout=120,
            )

            content = response.choices[0].message.content or ""
            # Strip markdown fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            result = json.loads(content)
            logger.info(
                "llm_judge_success",
                report_id=report_id,
                run_index=run_index,
                attempt=attempt,
            )
            return result

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                "llm_judge_invalid_json",
                report_id=report_id,
                run_index=run_index,
                attempt=attempt,
                error=str(e),
            )
        except Exception as e:
            last_error = e
            logger.error(
                "llm_judge_error",
                report_id=report_id,
                run_index=run_index,
                attempt=attempt,
                error=str(e),
            )
            raise

    raise ValueError(
        f"LLM judge failed to return valid JSON after {1 + max_retries} attempts: "
        f"{last_error}"
    )
