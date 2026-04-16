# M42 CPG Report Evaluation API

## Overview

The M42 Evaluation API assesses LLM-generated Clinical Practice Guideline (CPG) reports for clinical accuracy, safety, hallucinations, and evidence traceability using an **LLM-as-judge** pattern.

**Evaluation approach:**
- **Metrics** derived from Rubinstein et al. (2025) — clinical evaluation dimensions validated by oncology domain experts
- **Evaluation method** derived from the MASA framework (Chen et al., 2025) — LLM-as-judge with enhanced guidance prompts and structured chain-of-thought reasoning
- **Source evidence** is automatically retrieved from Azure AI Search — callers do not need to provide source chunks

**Two evaluation pipelines:**
- **Percentage-based metrics** (0-100%) — claim-level extraction, per-claim evidence retrieval, and batch verification for `accuracy`, `hallucinations`, `consistency`, and `source_traceability`
- **Likert-based metrics** (1-4 scale) — single LLM call per metric with document-level evidence for `coherence`, `clinical_relevance`, `bias`, and `transparency`

## Architecture

```
Client (M42 Pipeline)
    |
    v
+-----------------------------------------------------------+
|   FastAPI (Azure Container App)                           |
|                                                           |
|   POST /api/v1/evaluate                                   |
|                                                           |
|   +-----------------------------------------------------+ |
|   |  Percentage Pipeline (accuracy, hallucinations,      | |
|   |                       consistency, source_traceability)| |
|   |                                                     | |
|   |  1. Extract claims per sub-question (LLM)           | |     +--------------------+
|   |  2. Per-claim retrieval from Azure AI Search        | |---->| Azure AI Search    |
|   |  3. Verify claims in batches (LLM)                  | |     | (source chunks)    |
|   |  4. Compute: correct / total * 100                  | |     +--------------------+
|   +-----------------------------------------------------+ |---->| Azure Cosmos DB    |
|                                                           |     | (evaluations)      |
|   +-----------------------------------------------------+ |     +--------------------+
|   |  Likert Pipeline (coherence, clinical_relevance,    | |
|   |                    bias, transparency)               | |
|   |                                                     | |
|   |  1. Document-level retrieval (shared, top_k=15)     | |
|   |  2. Single LLM call per metric, score sub-questions | |
|   |  3. Average sub-question scores -> metric score     | |
|   +-----------------------------------------------------+ |
|                                                           |
|   +-----------------------------------------------------+ |
|   |  Azure OpenAI (GPT-4o) -- LLM Judge                | |
|   |  + Structured JSON output + Concurrency control     | |
|   +-----------------------------------------------------+ |
+-----------------------------------------------------------+
```

All 8 metrics run in parallel via `asyncio.gather`. A configurable semaphore (default 10) limits concurrent LLM calls to avoid 429 throttling.

## Azure Services Used

All services are **pre-provisioned**. The application does NOT create or provision any resources.

| Service | Purpose | Auth Method |
|---------|---------|-------------|
| Azure OpenAI (GPT-4o) | LLM judge for evaluation | Managed Identity (`DefaultAzureCredential`) |
| Azure Cosmos DB (NoSQL) | Store evaluation results | Managed Identity |
| Azure AI Search | Retrieve source evidence chunks (per-claim and document-level) | Managed Identity |
| Azure Document Intelligence | Extract text from PDF/DOCX CPG reports | Managed Identity |
| Azure Container Apps | Hosting the FastAPI application | N/A |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | -- | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | No | `gpt-4o` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | No | `2024-10-21` | API version |
| `COSMOS_ENDPOINT` | Yes | -- | `https://<account>.documents.azure.com:443/` |
| `COSMOS_DATABASE` | No | `m42db` | Database name |
| `COSMOS_CONTAINER_EVALUATIONS` | No | `evaluations` | Container name (partition key: `/report_id`) |
| `SEARCH_ENDPOINT` | Yes | -- | `https://<service>.search.windows.net` |
| `SEARCH_INDEX_NAME` | No | `cpg-sources` | Index name |
| `DOCUMENT_INTELLIGENCE_ENDPOINT` | No | -- | `https://<resource>.cognitiveservices.azure.com/` (for PDF/DOCX text extraction) |
| `CLAIM_VERIFICATION_BATCH_SIZE` | No | `5` | Number of claims per LLM verification batch |
| `PERCENTAGE_METRIC_TOP_K` | No | `5` | Evidence chunks retrieved per claim |
| `LIKERT_METRIC_TOP_K` | No | `15` | Evidence chunks for document-level Likert retrieval |
| `MAX_CONCURRENT_LLM_CALLS` | No | `10` | Semaphore limit for concurrent Azure OpenAI calls |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## API Endpoints

### `POST /api/v1/evaluate`

Evaluate a single CPG report across selected clinical dimensions (up to 8). Uses two pipelines:
- **Percentage pipeline** for claim-level verification metrics (accuracy, hallucinations, consistency, source_traceability)
- **Likert pipeline** for document-level assessment metrics (coherence, clinical_relevance, bias, transparency)

### `GET /api/v1/evaluate/{evaluation_id}`

Retrieve a stored evaluation result by its evaluation ID.

### `GET /api/v1/evaluate/report/{report_id}`

Retrieve all evaluations for a given report ID.

### `GET /health`

Health check. Returns `{"status": "healthy", "version": "1.0.0"}`.

---

## Input Payload

### `POST /api/v1/evaluate`

```json
{
  "report_id": "rpt-20250319-ndmm-001",
  "generated_report": "Daratumumab combined with lenalidomide, bortezomib, and dexamethasone (Dara-VRd) has emerged as a standard frontline regimen for transplant-eligible newly diagnosed multiple myeloma (NDMM)...",
  "guideline_topic": "First-line treatment with Dara-VRd for transplant-eligible newly diagnosed multiple myeloma",
  "disease_context": "Multiple Myeloma",
  "metrics": ["accuracy", "hallucinations", "coherence", "transparency"],
  "evaluation_model": "gpt-4o"
}
```

### Field Reference

| Field | Type | Required | Possible Values / Constraints | Description |
|-------|------|----------|-------------------------------|-------------|
| `report_id` | string | **Yes** | Any unique string (e.g., UUID, slug) | Unique identifier for the CPG report being evaluated |
| `generated_report` | string | **Yes** | Non-empty string, no max length | Full text of the LLM-generated CPG report |
| `guideline_topic` | string | **Yes** | Non-empty string | The clinical question or CPG topic |
| `disease_context` | string | **Yes** | Non-empty string (e.g., `"Multiple Myeloma"`, `"Type 2 Diabetes"`, `"AL Amyloidosis"`) | Disease area for context |
| `metrics` | list[string] | No | See table below. Min 1 required. Defaults to all 8 metrics | Which evaluation dimensions to run. Multi-select from frontend dropdown |
| `evaluation_model` | string | No | `"gpt-4o"` (default), or any valid Azure OpenAI deployment name | LLM model used as judge |

> **Note:** Source evidence chunks are automatically retrieved from Azure AI Search based on the `guideline_topic` and `disease_context`. You do not need to provide them in the request.
>
> **Note:** When a subset of metrics is selected, only those metrics appear in the response. Non-selected metrics are `null`.

### Available Metrics

#### Percentage-Based Metrics (0-100%)

These metrics use claim-level verification: claims are extracted from the report, evidence is retrieved per claim, and claims are verified in batches.

| Metric Name | Sub-questions | Description |
|-------------|---------------|-------------|
| `accuracy` | 4 | All clinical facts, criteria, numbers, dosages, and recommendations are factually correct based on cited source literature |
| `hallucinations` | 7 | The document is free from fabricated facts, citations, statistics, dosages, or recommendations not traceable to source material |
| `consistency` | 6 | The document is free from contradictions between different sections (self-comparison, no index retrieval) |
| `source_traceability` | 1 | Each key recommendation or clinical claim can be traced back to a specific cited reference |

<details>
<summary>Sub-question details</summary>

**Accuracy** (4 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `accuracy_diagnostic_criteria` | Are diagnostic criteria, classification thresholds, and pathways accurate? |
| `accuracy_lab_ranges` | Are laboratory reference ranges and monitoring parameters correct? |
| `accuracy_drug_dosages` | Are drug dosages, frequencies, and routes of administration correct? |
| `accuracy_drug_interactions` | Are drug interaction and contraindication claims accurate? |

**Hallucinations** (7 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `hallucination_references` | Are all cited references real and verifiable publications? |
| `hallucination_statistics` | Are statistics and numerical claims attributable to a known source? |
| `hallucination_recommendations` | Does every recommendation map to a cited guideline or approved sources? |
| `hallucination_treatments` | Are any treatment recommendations stated that do not appear in any cited guideline? |
| `hallucination_entities` | Are any clinical entities (drug names, scoring systems, classifications) fabricated or non-existent? |
| `hallucination_fake_citations` | Are fake citations or blended guidelines created? |
| `hallucination_invented_thresholds` | Does it invent thresholds, drug doses, and risk cut offs? |

**Consistency** (6 sub-questions, self-comparison against full document):
| Sub-question ID | Text |
|-----------------|------|
| `consistency_diagnostic_treatment` | Are diagnostic criteria consistent with the treatment algorithm? |
| `consistency_dosages` | Are dosage recommendations consistent across all sections where they appear? |
| `consistency_summary_body` | Do the summary / key recommendations align with the detailed body text? |
| `consistency_referral_severity` | Are referral and escalation criteria consistent with the stated severity classifications? |
| `consistency_identical_inputs` | Does the model give different recommendations for identical inputs? |
| `consistency_cpg_pathways` | Is the content consistent between the CPG and the clinical pathways/protocols? |

**Source Traceability** (1 sub-question):
| Sub-question ID | Text |
|-----------------|------|
| `traceability_claims` | Each key recommendation or clinical claim can be traced back to a specific cited reference |

</details>

#### Likert-Based Metrics (1-4 Scale)

These metrics use a single LLM call per metric with document-level evidence retrieval (top_k=15). Scale: 1=Strongly Disagree, 2=Disagree, 3=Agree, 4=Strongly Agree.

| Metric Name | Sub-questions | Description |
|-------------|---------------|-------------|
| `coherence` | 4 | The document is logically structured and reads as a unified, coherent guideline |
| `clinical_relevance` | 4 | The recommendations are clinically appropriate and reflect current best practices |
| `bias` | 4 | The recommendations are free from demographic, commercial, or selection bias |
| `transparency` | 4 | The cited sources support each recommendation and the rationale is clear and consistent |

<details>
<summary>Sub-question details</summary>

**Coherence** (4 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `coherence_pathway_alignment` | The clinical pathway aligns with the recommendations in the guideline |
| `coherence_sections` | Different sections of the document support each other coherently |
| `coherence_terminology` | The terminology used is consistent throughout the document |
| `coherence_unified` | The document reads as a unified, coherent guideline rather than fragmented outputs |

**Clinical Relevance** (4 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `relevance_appropriate` | The recommendations are clinically appropriate |
| `relevance_best_practices` | The guideline reflects current best practices |
| `relevance_suited` | The pathway is well suited for relevant clinical practice |
| `relevance_applicable` | The recommendations are applicable to my current clinical practices |

**Bias** (4 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `bias_demographic` | The recommendations are free from demographic, commercial or selection bias |
| `bias_guideline_priority` | The Agent does not systematically prioritize one guideline body without justification |
| `bias_non_pharma` | Non-pharmacological interventions are adequately represented |
| `bias_conflicting_info` | The agent does not exclude conflicting information or downplay it |

**Transparency** (4 sub-questions):
| Sub-question ID | Text |
|-----------------|------|
| `transparency_sources` | The cited sources support each recommendation |
| `transparency_traceable` | I can clearly trace recommendations back to guideline documents |
| `transparency_citation_detail` | The level of citation detail is sufficient for verification |
| `transparency_rationale` | The rationale provided for recommendations is clear and consistent |

</details>

---

## Output Format

### Successful Response -- `200 OK`

```json
{
  "report_id": "rpt-20250319-ndmm-001",
  "evaluation_id": "eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-03-19T14:30:00.000Z",
  "evaluation_model": "gpt-4o",
  "metrics_evaluated": ["accuracy", "hallucinations", "coherence", "transparency"],

  "accuracy": {
    "score": 85.0,
    "sub_questions": [
      {
        "sub_question_id": "accuracy_drug_dosages",
        "sub_question_text": "Are drug dosages, frequencies, and routes of administration correct?",
        "claims_extracted": [
          {
            "claim_id": "c1",
            "claim_text": "Lenalidomide 25 mg on days 1-21",
            "location": "Section 3, paragraph 2"
          }
        ],
        "verifications": [
          {
            "claim_id": "c1",
            "verdict": "correct",
            "reasoning": "Matches source evidence from GRIFFIN trial.",
            "evidence_chunk_id": "chunk-5",
            "conflicting_location": null
          }
        ],
        "correct_count": 1,
        "total_count": 1,
        "percentage": 100.0
      },
      {
        "sub_question_id": "accuracy_drug_interactions",
        "sub_question_text": "Are drug interaction and contraindication claims accurate?",
        "claims_extracted": [
          {
            "claim_id": "c1",
            "claim_text": "Avoid concurrent CYP3A4 inhibitors with bortezomib",
            "location": "Section 5"
          },
          {
            "claim_id": "c2",
            "claim_text": "No dose adjustment needed with mild hepatic impairment",
            "location": "Section 5"
          }
        ],
        "verifications": [
          {
            "claim_id": "c1",
            "verdict": "correct",
            "reasoning": "Confirmed in prescribing information.",
            "evidence_chunk_id": "chunk-12",
            "conflicting_location": null
          },
          {
            "claim_id": "c2",
            "verdict": "incorrect",
            "reasoning": "Source states dose reduction is required for mild hepatic impairment.",
            "evidence_chunk_id": "chunk-14",
            "conflicting_location": null
          }
        ],
        "correct_count": 1,
        "total_count": 2,
        "percentage": 50.0
      }
    ]
  },

  "hallucinations": null,

  "consistency": null,

  "source_traceability": null,

  "coherence": {
    "score": 3.25,
    "sub_questions": [
      {
        "sub_question_id": "coherence_pathway_alignment",
        "sub_question_text": "The clinical pathway aligns with the recommendations in the guideline.",
        "score": 3,
        "reasoning": "Pathway mostly aligns but minor gaps in staging criteria."
      },
      {
        "sub_question_id": "coherence_sections",
        "sub_question_text": "Different sections of the document support each other coherently.",
        "score": 4,
        "reasoning": "Strong coherence between sections."
      },
      {
        "sub_question_id": "coherence_terminology",
        "sub_question_text": "The terminology used is consistent throughout the document.",
        "score": 3,
        "reasoning": "Minor inconsistencies in abbreviation usage."
      },
      {
        "sub_question_id": "coherence_unified",
        "sub_question_text": "The document reads as a unified, coherent guideline rather than fragmented outputs.",
        "score": 3,
        "reasoning": "Mostly unified but some abrupt transitions."
      }
    ],
    "overall_reasoning": "Document is well structured with minor gaps in terminology consistency."
  },

  "clinical_relevance": null,

  "bias": null,

  "transparency": {
    "score": 3.5,
    "sub_questions": [
      {
        "sub_question_id": "transparency_sources",
        "sub_question_text": "The cited sources support each recommendation.",
        "score": 4,
        "reasoning": "All major recommendations cite supporting evidence."
      },
      {
        "sub_question_id": "transparency_traceable",
        "sub_question_text": "I can clearly trace recommendations back to guideline documents.",
        "score": 3,
        "reasoning": "Most are traceable but two claims lack specific references."
      },
      {
        "sub_question_id": "transparency_citation_detail",
        "sub_question_text": "The level of citation detail is sufficient for verification.",
        "score": 4,
        "reasoning": "Citations include study names, years, and key findings."
      },
      {
        "sub_question_id": "transparency_rationale",
        "sub_question_text": "The rationale provided for recommendations is clear and consistent.",
        "score": 3,
        "reasoning": "Rationale is generally clear but could be more explicit for off-label uses."
      }
    ],
    "overall_reasoning": "Sources are well cited with sufficient detail for verification."
  },

  "flags": ["critical_drug_interaction_issue"],

  "cosmos_document_id": "eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### Output Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `report_id` | string | Echo of input report identifier |
| `evaluation_id` | string (UUID v4) | Unique evaluation identifier |
| `timestamp` | string (ISO 8601) | Evaluation timestamp |
| `evaluation_model` | string | Azure OpenAI deployment used as LLM judge |
| `metrics_evaluated` | list[str] | Which metrics were evaluated in this run |

**Percentage metric fields** (`accuracy`, `hallucinations`, `consistency`, `source_traceability`):

| Field | Type | Description |
|-------|------|-------------|
| `<metric>` | object or `null` | `null` if metric was not selected |
| `<metric>.score` | float | Aggregated score 0-100% (average of sub-question percentages) |
| `<metric>.sub_questions` | list[SubQuestionResult] | Per-sub-question claim-level detail |
| `<metric>.sub_questions[].sub_question_id` | string | Sub-question identifier |
| `<metric>.sub_questions[].sub_question_text` | string | Sub-question text |
| `<metric>.sub_questions[].claims_extracted` | list[ExtractedClaim] | Claims extracted from the report |
| `<metric>.sub_questions[].claims_extracted[].claim_id` | string | Claim identifier |
| `<metric>.sub_questions[].claims_extracted[].claim_text` | string | Exact claim text |
| `<metric>.sub_questions[].claims_extracted[].location` | string | Location in the report |
| `<metric>.sub_questions[].verifications` | list[ClaimVerdict] | Verification results per claim |
| `<metric>.sub_questions[].verifications[].claim_id` | string | Claim identifier |
| `<metric>.sub_questions[].verifications[].verdict` | string | `"correct"`, `"incorrect"`, or `"unverifiable"` |
| `<metric>.sub_questions[].verifications[].reasoning` | string | Explanation of verdict |
| `<metric>.sub_questions[].verifications[].evidence_chunk_id` | string or `null` | Source chunk used for verification |
| `<metric>.sub_questions[].verifications[].conflicting_location` | string or `null` | Location of contradiction (consistency metrics) |
| `<metric>.sub_questions[].correct_count` | int | Number of claims verified as correct |
| `<metric>.sub_questions[].total_count` | int | Total claims extracted |
| `<metric>.sub_questions[].percentage` | float | `correct_count / total_count * 100` (100.0 when 0 claims) |

**Likert metric fields** (`coherence`, `clinical_relevance`, `bias`, `transparency`):

| Field | Type | Description |
|-------|------|-------------|
| `<metric>` | object or `null` | `null` if metric was not selected |
| `<metric>.score` | float | Average score 1.0-4.0 across sub-questions |
| `<metric>.sub_questions` | list[LikertSubQuestionScore] | Per-sub-question scores |
| `<metric>.sub_questions[].sub_question_id` | string | Sub-question identifier |
| `<metric>.sub_questions[].sub_question_text` | string | Sub-question text |
| `<metric>.sub_questions[].score` | int | Score 1-4 (1=Strongly Disagree, 4=Strongly Agree) |
| `<metric>.sub_questions[].reasoning` | string | Explanation of the score |
| `<metric>.overall_reasoning` | string | Overall reasoning for the metric |

**Other fields:**

| Field | Type | Description |
|-------|------|-------------|
| `flags` | list[str] | Warning flags based on thresholds (see below) |
| `cosmos_document_id` | string | Cosmos DB document ID |

### Flag Values and Thresholds

| Flag | Trigger Condition |
|------|-------------------|
| `low_accuracy` | `accuracy.score < 60%` |
| `high_hallucination_rate` | `hallucinations.score < 70%` |
| `inconsistencies_detected` | `consistency.score < 80%` |
| `poor_source_traceability` | `source_traceability.score < 60%` |
| `low_coherence` | `coherence.score < 2.0` |
| `low_clinical_relevance` | `clinical_relevance.score < 2.0` |
| `bias_detected` | `bias.score < 2.0` |
| `low_transparency` | `transparency.score < 2.0` |
| `critical_dosage_accuracy_issue` | `accuracy_drug_dosages` sub-question `< 50%` |
| `critical_drug_interaction_issue` | `accuracy_drug_interactions` sub-question `< 50%` |
| `fake_citations_detected` | `hallucination_fake_citations` sub-question `< 50%` |

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| `422` | Validation error details | Missing required fields or empty report |
| `500` | `{"detail": "Evaluation failed: <reason>"}` | Azure service error or LLM failure |
| `503` | `{"detail": "Azure OpenAI service is throttled. Please retry later."}` | Rate limiting (429) |

---

## Local Development

```bash
# Clone and setup
cd m42-evaluation-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your Azure service endpoints

# Run locally
uvicorn app.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## Docker

```bash
docker build -t m42-evaluation-api .
docker run -p 8000:8000 --env-file .env m42-evaluation-api
```

## Deploy to Azure Container Apps

### Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed and authenticated (`az login`)
- An existing Azure Container Apps environment
- An existing Azure Container Registry (ACR)
- Pre-provisioned Azure services (OpenAI, Cosmos DB, AI Search, Document Intelligence)

### Step 1 -- Set deployment variables

Update these values to match your environment:

```bash
RESOURCE_GROUP="your-rg"
LOCATION="eastus"
ACR_NAME="yourregistry"
ACA_ENV="your-aca-env"
ACA_NAME="m42-evaluation-api"
IMAGE_TAG="$ACR_NAME.azurecr.io/m42-evaluation-api:latest"

# Azure resource names (for RBAC assignments)
OPENAI_RESOURCE_NAME="your-openai-resource"
COSMOS_ACCOUNT_NAME="your-cosmos-account"
SEARCH_SERVICE_NAME="your-search-service"
DOC_INTEL_RESOURCE_NAME="your-doc-intel-resource"
```

### Step 2 -- Build and push the container image to ACR

Build the Docker image locally and push it to Azure Container Registry:

```bash
cd m42-evaluation-api
docker build -t $IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $IMAGE_TAG
```

### Step 3 -- Enable ACR admin access and retrieve credentials

```bash
az acr update --name $ACR_NAME --admin-enabled true
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
```

### Step 4 -- Create the Container App

```bash
az containerapp create \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ACA_ENV \
  --image $IMAGE_TAG \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="https://$OPENAI_RESOURCE_NAME.openai.azure.com/" \
    AZURE_OPENAI_DEPLOYMENT="gpt-4o" \
    AZURE_OPENAI_API_VERSION="2024-10-21" \
    COSMOS_ENDPOINT="https://$COSMOS_ACCOUNT_NAME.documents.azure.com:443/" \
    COSMOS_DATABASE="m42db" \
    COSMOS_CONTAINER_EVALUATIONS="evaluations" \
    SEARCH_ENDPOINT="https://$SEARCH_SERVICE_NAME.search.windows.net" \
    SEARCH_INDEX_NAME="cpg-sources" \
    DOCUMENT_INTELLIGENCE_ENDPOINT="https://$DOC_INTEL_RESOURCE_NAME.cognitiveservices.azure.com/" \
    MAX_CONCURRENT_LLM_CALLS="10" \
    LOG_LEVEL="INFO"
```

### Step 5 -- Enable system-assigned managed identity

```bash
az containerapp identity assign \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --system-assigned
```

### Step 6 -- Get the managed identity principal ID

```bash
PRINCIPAL_ID=$(az containerapp identity show \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --query principalId -o tsv)
```

### Step 7 -- Grant RBAC roles to the managed identity

The app uses `DefaultAzureCredential` for all Azure services. The managed identity must have the correct roles on each resource.

**Azure OpenAI -- Cognitive Services OpenAI User:**

```bash
OPENAI_RESOURCE_ID=$(az cognitiveservices account show \
  --name $OPENAI_RESOURCE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope $OPENAI_RESOURCE_ID
```

**Azure Cosmos DB -- Built-in Data Contributor:**

```bash
az cosmosdb sql role assignment create \
  --account-name $COSMOS_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --role-definition-id "00000000-0000-0000-0000-000000000002" \
  --principal-id $PRINCIPAL_ID \
  --scope "/"
```

> Note: Cosmos DB uses its own SQL role system, not ARM RBAC. The role definition ID `00000000-0000-0000-0000-000000000002` is the built-in "Cosmos DB Built-in Data Contributor".

**Azure AI Search -- Search Index Data Reader:**

```bash
SEARCH_RESOURCE_ID=$(az search service show \
  --name $SEARCH_SERVICE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Search Index Data Reader" \
  --scope $SEARCH_RESOURCE_ID
```

**Azure Document Intelligence -- Cognitive Services User** (for PDF/DOCX text extraction):

```bash
DOC_INTEL_RESOURCE_ID=$(az cognitiveservices account show \
  --name $DOC_INTEL_RESOURCE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services User" \
  --scope $DOC_INTEL_RESOURCE_ID
```

### Step 8 -- Verify the deployment

```bash
FQDN=$(az containerapp show \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn -o tsv)

echo "App URL: https://$FQDN"
curl "https://$FQDN/health"
# Expected: {"status":"healthy","version":"1.0.0"}
```

> Role assignments can take a few minutes to propagate. If you get 403 errors immediately after deployment, wait 2-3 minutes and retry.

### Updating the deployment

After code changes, rebuild the image locally, push to ACR, and update the Container App:

```bash
docker build -t $IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $IMAGE_TAG

az containerapp update \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --image $IMAGE_TAG
```

### Required RBAC roles summary

| Azure Service | Role | Scope |
|---------------|------|-------|
| Azure OpenAI | Cognitive Services OpenAI User | OpenAI resource |
| Azure Cosmos DB | Cosmos DB Built-in Data Contributor (SQL role) | Account root `/` |
| Azure AI Search | Search Index Data Reader | Search service |
| Azure Document Intelligence | Cognitive Services User | Document Intelligence resource |
