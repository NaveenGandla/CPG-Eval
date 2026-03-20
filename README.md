# M42 CPG Report Evaluation API

## Overview

The M42 Evaluation API assesses LLM-generated Clinical Practice Guideline (CPG) reports for clinical accuracy, safety, hallucinations, and evidence traceability using an **LLM-as-judge** pattern with majority voting.

**Evaluation approach:**
- **Metrics** derived from Rubinstein et al. (2025) — clinical evaluation dimensions validated by oncology domain experts
- **Evaluation method** derived from the MASA framework (Chen et al., 2025) — LLM-as-judge with enhanced guidance prompts, majority voting across multiple independent runs, and structured chain-of-thought reasoning

## Architecture

```
Client (M42 Pipeline)
    │
    ▼
┌──────────────────────────────────┐
│   FastAPI (Azure Container App)  │
│                                  │
│   POST /api/v1/evaluate          │
│                                  │
│   ┌────────────────────────────┐ │
│   │   Evaluation Engine        │ │
│   │                            │ │
│   │   Run 1 ─┐                 │ │
│   │   Run 2 ─┼─► Aggregate    │ │     ┌──────────────────┐
│   │   Run 3 ─┘   (median +    │ │────►│ Azure Cosmos DB  │
│   │               majority)    │ │     │ (evaluations)    │
│   └────────────────────────────┘ │     └──────────────────┘
│              │                   │
│              ▼                   │     ┌──────────────────┐
│   ┌────────────────────────────┐ │────►│ Azure Blob Store │
│   │   Azure OpenAI (GPT-4o)   │ │     │ (full reports)   │
│   │   LLM Judge                │ │     └──────────────────┘
│   └────────────────────────────┘ │
│              │                   │     ┌──────────────────┐
│              ▼                   │────►│ Azure AI Search  │
│   ┌────────────────────────────┐ │     │ (source chunks)  │
│   │   Enhanced Guidance Prompt │ │     └──────────────────┘
│   │   + Chain-of-Thought       │ │
│   └────────────────────────────┘ │
└──────────────────────────────────┘
```

## Azure Services Used

All services are **pre-provisioned**. The application does NOT create or provision any resources.

| Service | Purpose | Auth Method |
|---------|---------|-------------|
| Azure OpenAI (GPT-4o) | LLM judge for evaluation | Managed Identity (`DefaultAzureCredential`) |
| Azure Cosmos DB (NoSQL) | Store evaluation results | Managed Identity |
| Azure AI Search | Retrieve/enrich source chunks | Managed Identity |
| Azure Blob Storage | Store full evaluation report JSONs | Managed Identity |
| Azure Container Apps | Hosting the FastAPI application | N/A |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | — | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | No | `gpt-4o` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | No | `2024-10-21` | API version |
| `COSMOS_ENDPOINT` | Yes | — | `https://<account>.documents.azure.com:443/` |
| `COSMOS_DATABASE` | No | `m42db` | Database name |
| `COSMOS_CONTAINER_EVALUATIONS` | No | `evaluations` | Container name (partition key: `/report_id`) |
| `SEARCH_ENDPOINT` | Yes | — | `https://<service>.search.windows.net` |
| `SEARCH_INDEX_NAME` | No | `cpg-sources` | Index name |
| `BLOB_ACCOUNT_URL` | Yes | — | `https://<account>.blob.core.windows.net` |
| `BLOB_CONTAINER_NAME` | No | `evaluation-reports` | Blob container name |
| `DEFAULT_NUM_EVAL_RUNS` | No | `3` | Default number of independent evaluation runs |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## API Endpoints

### `POST /api/v1/evaluate`

Evaluate a single CPG report across 8 clinical dimensions.

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
  "retrieved_chunks": [
    {
      "chunk_id": "chunk-001",
      "text": "The PERSEUS trial (Sonneveld et al., NEJM 2024) demonstrated that Dara-VRd significantly improved progression-free survival compared to VRd alone in transplant-eligible NDMM patients...",
      "metadata": {
        "study_name": "PERSEUS",
        "year": 2024,
        "journal": "New England Journal of Medicine",
        "authors": "Sonneveld P, Dimopoulos MA, Boccadoro M, et al."
      }
    },
    {
      "chunk_id": "chunk-002",
      "text": "The GRIFFIN trial (Chari et al., Blood Cancer J 2024) evaluated Dara-VRd in transplant-eligible NDMM...",
      "metadata": {
        "study_name": "GRIFFIN",
        "year": 2024,
        "journal": "Blood Cancer Journal",
        "authors": "Chari A, Kaufman JL, Laubach J, et al."
      }
    }
  ],
  "guideline_topic": "First-line treatment with Dara-VRd for transplant-eligible newly diagnosed multiple myeloma",
  "disease_context": "Multiple Myeloma",
  "reference_report": null,
  "evaluation_model": "gpt-4o",
  "num_eval_runs": 3
}
```

### Field Reference

| Field | Type | Required | Possible Values / Constraints | Description |
|-------|------|----------|-------------------------------|-------------|
| `report_id` | string | **Yes** | Any unique string (e.g., UUID, slug) | Unique identifier for the CPG report being evaluated |
| `generated_report` | string | **Yes** | Non-empty string, no max length | Full text of the LLM-generated CPG report |
| `retrieved_chunks` | list[SourceChunk] | **Yes** | Min 1 chunk required | Source evidence chunks from Azure AI Search used during report generation |
| `retrieved_chunks[].chunk_id` | string | **Yes** | Any unique string | Identifier for the chunk |
| `retrieved_chunks[].text` | string | **Yes** | Non-empty string | Full text content of the chunk |
| `retrieved_chunks[].metadata.study_name` | string | No | Any string or `null` | Name of the clinical trial or study |
| `retrieved_chunks[].metadata.year` | int | No | 1900–2030 or `null` | Publication year |
| `retrieved_chunks[].metadata.journal` | string | No | Any string or `null` | Journal name |
| `retrieved_chunks[].metadata.authors` | string | No | Any string or `null` | Author list |
| `guideline_topic` | string | **Yes** | Non-empty string | The clinical question or CPG topic |
| `disease_context` | string | **Yes** | Non-empty string (e.g., `"Multiple Myeloma"`, `"Type 2 Diabetes"`, `"AL Amyloidosis"`) | Disease area for context |
| `reference_report` | string | No | Any string or `null` | Gold-standard manually curated report for comparison (future use) |
| `evaluation_model` | string | No | `"gpt-4o"` (default), or any valid Azure OpenAI deployment name | LLM model used as judge |
| `num_eval_runs` | int | No | `1` to `7`, default `3` | Number of independent evaluation runs for majority voting |

---

## Output Format

### Successful Response — `200 OK`

```json
{
  "report_id": "rpt-20250319-ndmm-001",
  "evaluation_id": "eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-03-19T14:30:00.000Z",
  "model_used": "gpt-4o",
  "num_runs": 3,

  "clinical_accuracy": {
    "score": 4,
    "confidence": "high",
    "reasoning": "Drug names and trial outcomes are correctly stated. Minor imprecision in PFS hazard ratio (0.42 stated vs 0.44 in source) does not alter clinical interpretation."
  },
  "completeness": {
    "score": 3,
    "confidence": "medium",
    "reasoning": "Key trials PERSEUS and GRIFFIN are covered. However, subgroup analyses for elderly patients and the maintenance therapy phase are not addressed."
  },
  "safety_completeness": {
    "score": 2,
    "confidence": "high",
    "reasoning": "Report mentions 'generally well-tolerated' without specific adverse event rates. No mention of infusion reactions, neutropenia rates, or cardiac monitoring requirements.",
    "missing_items": [
      "Infusion-related reaction rates (Grade 3/4)",
      "Neutropenia incidence and G-CSF requirements",
      "Cardiac monitoring for bortezomib-associated cardiotoxicity",
      "Peripheral neuropathy grading and dose modification criteria",
      "VTE prophylaxis requirements with lenalidomide"
    ]
  },
  "relevance": {
    "score": 5,
    "confidence": "high",
    "reasoning": "All content is directly pertinent to frontline Dara-VRd in transplant-eligible NDMM. No tangential information detected."
  },
  "coherence": {
    "score": 4,
    "confidence": "high",
    "reasoning": "Well-structured with clear progression from trial rationale to efficacy data to clinical implications. Minor abrupt transition between GRIFFIN and PERSEUS sections."
  },
  "evidence_traceability": {
    "score": 3,
    "confidence": "medium",
    "reasoning": "PERSEUS and GRIFFIN trial data are attributable to source chunks. Two claims about MRD negativity rates lack specific source attribution.",
    "untraced_claims": [
      {
        "claim": "MRD negativity was achieved in 75.2% of patients at 12 months",
        "location": "paragraph 3"
      },
      {
        "claim": "Dara-VRd is now considered category 1 preferred by NCCN",
        "location": "paragraph 5"
      }
    ]
  },
  "hallucination_score": {
    "score": 3,
    "confidence": "medium",
    "reasoning": "One minor hallucination detected — the stated MRD negativity rate does not match any source chunk. Core clinical conclusions are grounded in evidence."
  },
  "fih_detected": [
    {
      "claim": "The GRIFFIN trial enrolled 207 patients",
      "source_says": "GRIFFIN enrolled 104 patients in the Dara-VRd arm and 103 in VRd arm (total 207 randomized but source chunk states 'over 200 patients' without the exact number)",
      "severity": "minor",
      "location": "paragraph 2"
    }
  ],

  "overall_score": 68.75,
  "usable_without_editing": false,
  "confidence_level": "medium",
  "flags": [
    "missing_safety_data",
    "untraced_claims_detected"
  ],

  "cosmos_document_id": "eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "blob_url": "https://<account>.blob.core.windows.net/evaluation-reports/rpt-20250319-ndmm-001/eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890.json"
}
```

### Output Field Reference

| Field | Type | Possible Values | Description |
|-------|------|-----------------|-------------|
| `report_id` | string | Echo of input | Report identifier |
| `evaluation_id` | string | UUID v4 | Unique evaluation identifier |
| `timestamp` | string | ISO 8601 | Evaluation timestamp |
| `model_used` | string | `"gpt-4o"` or deployment name | LLM judge model used |
| `num_runs` | int | 1–7 | Number of evaluation runs performed |
| `clinical_accuracy.score` | int | `1`, `2`, `3`, `4`, `5` | 1=pervasive inaccuracies, 5=all facts correct |
| `completeness.score` | int | `1`, `2`, `3`, `4`, `5` | 1=majority missing, 5=all critical info included |
| `safety_completeness.score` | int | `1`, `2`, `3`, `4`, `5` | 1=absent/dangerous, 5=comprehensive with AE rates |
| `safety_completeness.missing_items` | list[str] | Any safety-related items | Specific safety data points missing from the report |
| `relevance.score` | int | `1`, `2`, `3`, `4`, `5` | 1=mostly irrelevant, 5=perfectly on-topic |
| `coherence.score` | int | `1`, `2`, `3`, `4`, `5` | 1=incoherent, 5=professional clinical document |
| `evidence_traceability.score` | int | `1`, `2`, `3`, `4`, `5` | 1=most claims unattributed, 5=every claim traceable |
| `evidence_traceability.untraced_claims` | list[object] | `{claim, location}` | Claims that cannot be traced to source chunks |
| `hallucination_score.score` | int | `1`, `2`, `3`, `4` | 1=many hallucinations, 4=none detected |
| `fih_detected` | list[FIHItem] | See below | Specific factually incorrect claims |
| `fih_detected[].claim` | string | Exact text | The incorrect claim from the report |
| `fih_detected[].source_says` | string | Evidence text | What the source actually states |
| `fih_detected[].severity` | string | `"critical"`, `"major"`, `"minor"` | Impact on patient safety |
| `fih_detected[].location` | string | e.g., `"paragraph 2"` | Where in the report |
| `overall_score` | float | `0.00` – `100.00` | Weighted aggregate score |
| `usable_without_editing` | bool | `true`, `false` | `true` if score >= 80 and no critical FIH |
| `confidence_level` | string | `"high"`, `"medium"`, `"low"` | Agreement level across evaluation runs |
| `flags` | list[str] | See below | Critical issues detected |

### Possible Flag Values

| Flag | Trigger Condition |
|------|-------------------|
| `missing_safety_data` | `safety_completeness.score <= 2` |
| `hallucinated_citation` | Any FIH with severity `"critical"` or `"major"` |
| `low_accuracy` | `clinical_accuracy.score <= 2` |
| `untraced_claims_detected` | `evidence_traceability.untraced_claims` is non-empty |
| `low_confidence` | `confidence_level == "low"` |
| `many_hallucinations` | `hallucination_score.score <= 2` |
| `incomplete_report` | `completeness.score <= 2` |

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| `400` | `{"detail": "generated_report cannot be empty"}` | Validation error |
| `400` | `{"detail": "retrieved_chunks must contain at least 1 chunk"}` | No source chunks |
| `500` | `{"detail": "Evaluation engine failed: <reason>"}` | Azure service error |
| `503` | `{"detail": "Azure OpenAI service throttled, retry after <N> seconds"}` | Rate limiting |

---

## Scoring Weights

| Metric | Weight | Justification |
|--------|--------|---------------|
| Clinical Accuracy | 25% | Highest clinical impact — wrong information is dangerous |
| Safety Completeness | 20% | LLMs systematically under-report safety data (Rubinstein 2025) |
| Evidence Traceability | 20% | Core RAG requirement — every claim needs a verifiable source |
| Hallucination Score | 15% | Fabricated content destroys trust and can harm patients |
| Completeness | 10% | Important but less critical than accuracy |
| Relevance | 5% | LLMs generally perform well on relevance |
| Coherence | 5% | LLMs generally perform well on coherence |

**Formula:** `overall_score = Σ(normalized_metric_score × weight) × 100`

- Likert 1–5 normalized: `(score - 1) / 4` → range 0.0 to 1.0
- Hallucination 1–4 normalized: `(score - 1) / 3` → range 0.0 to 1.0

---

## Routing Thresholds

| Condition | Action |
|-----------|--------|
| `overall_score >= 80` AND `fih_detected` is empty AND `usable_without_editing == true` | Auto-approve report |
| `overall_score >= 60` AND no critical flags | Route to expert for light review |
| `overall_score < 60` OR critical flags present | Route to expert for full review + consider regeneration |
| `hallucination_score <= 2` OR any FIH with `severity == "critical"` | **Block report**, trigger regeneration |

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
- Pre-provisioned Azure services (OpenAI, Cosmos DB, AI Search, Blob Storage)

### Step 1 — Set deployment variables

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
STORAGE_ACCOUNT_NAME="your-storage-account"
SEARCH_SERVICE_NAME="your-search-service"
```

### Step 2 — Build and push the container image to ACR

Build the Docker image locally and push it to Azure Container Registry:

```bash
cd m42-evaluation-api
docker build -t $IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $IMAGE_TAG
```

### Step 3 — Enable ACR admin access (if not already enabled)

```bash
az acr update --name $ACR_NAME --admin-enabled true
```

### Step 4 — Create the Container App

```bash
az containerapp create \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ACA_ENV \
  --image $IMAGE_TAG \
  --registry-server "$ACR_NAME.azurecr.io" \
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
    BLOB_ACCOUNT_URL="https://$STORAGE_ACCOUNT_NAME.blob.core.windows.net" \
    BLOB_CONTAINER_NAME="evaluation-reports" \
    LOG_LEVEL="INFO"
```

### Step 5 — Enable system-assigned managed identity

```bash
az containerapp identity assign \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --system-assigned
```

### Step 6 — Get the managed identity principal ID

```bash
PRINCIPAL_ID=$(az containerapp identity show \
  --name $ACA_NAME \
  --resource-group $RESOURCE_GROUP \
  --query principalId -o tsv)
```

### Step 7 — Grant RBAC roles to the managed identity

The app uses `DefaultAzureCredential` for all Azure services. The managed identity must have the correct roles on each resource.

**Azure OpenAI — Cognitive Services OpenAI User:**

```bash
OPENAI_RESOURCE_ID=$(az cognitiveservices account show \
  --name $OPENAI_RESOURCE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services OpenAI User" \
  --scope $OPENAI_RESOURCE_ID
```

**Azure Cosmos DB — Built-in Data Contributor:**

```bash
az cosmosdb sql role assignment create \
  --account-name $COSMOS_ACCOUNT_NAME \
  --resource-group $RESOURCE_GROUP \
  --role-definition-id "00000000-0000-0000-0000-000000000002" \
  --principal-id $PRINCIPAL_ID \
  --scope "/"
```

> Note: Cosmos DB uses its own SQL role system, not ARM RBAC. The role definition ID `00000000-0000-0000-0000-000000000002` is the built-in "Cosmos DB Built-in Data Contributor".

**Azure Blob Storage — Storage Blob Data Contributor:**

```bash
STORAGE_RESOURCE_ID=$(az storage account show \
  --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Contributor" \
  --scope $STORAGE_RESOURCE_ID
```

**Azure AI Search — Search Index Data Reader:**

```bash
SEARCH_RESOURCE_ID=$(az search service show \
  --name $SEARCH_SERVICE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Search Index Data Reader" \
  --scope $SEARCH_RESOURCE_ID
```

### Step 8 — Verify the deployment

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
| Azure Blob Storage | Storage Blob Data Contributor | Storage account |
| Azure AI Search | Search Index Data Reader | Search service |
