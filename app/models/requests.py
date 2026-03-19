from pydantic import BaseModel, Field


class SourceChunkMetadata(BaseModel):
    study_name: str | None = None
    year: int | None = None
    journal: str | None = None
    authors: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: SourceChunkMetadata = SourceChunkMetadata()


class EvaluationRequest(BaseModel):
    report_id: str = Field(..., description="Unique ID for the CPG report")
    generated_report: str = Field(
        ..., min_length=1, description="Full text of the generated report"
    )
    retrieved_chunks: list[SourceChunk] = Field(
        ..., min_length=1, description="Source chunks from Azure AI Search"
    )
    guideline_topic: str = Field(
        ...,
        min_length=1,
        description="e.g. 'First-line treatment for transplant-eligible NDMM'",
    )
    disease_context: str = Field(
        ..., min_length=1, description="e.g. 'Multiple Myeloma'"
    )
    reference_report: str | None = Field(
        default=None, description="Optional gold-standard report"
    )
    evaluation_model: str = Field(
        default="gpt-4o", description="Azure OpenAI deployment name"
    )
    num_eval_runs: int = Field(
        default=3, ge=1, le=7, description="Number of independent eval runs"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "report_id": "rpt-001",
                    "generated_report": "This CPG report covers first-line treatment options for newly diagnosed multiple myeloma...",
                    "retrieved_chunks": [
                        {
                            "chunk_id": "chunk-1",
                            "text": "The GRIFFIN trial demonstrated that D-VRd followed by...",
                            "metadata": {
                                "study_name": "GRIFFIN",
                                "year": 2023,
                                "journal": "Blood",
                            },
                        }
                    ],
                    "guideline_topic": "First-line treatment for transplant-eligible NDMM",
                    "disease_context": "Multiple Myeloma",
                    "num_eval_runs": 3,
                }
            ]
        }
    }
