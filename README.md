# M42 CPG Report Evaluation API

## Overview

The M42 Evaluation API assesses LLM-generated Clinical Practice Guideline (CPG) reports for clinical accuracy, safety, hallucinations, and evidence traceability using an **LLM-as-judge** pattern.

**Evaluation approach:**
- **Metrics** derived from Rubinstein et al. (2025) — clinical evaluation dimensions validated by oncology domain experts
- **Evaluation method** derived from the MASA framework (Chen et al., 2025) — LLM-as-judge with enhanced guidance prompts and structured chain-of-thought reasoning
- **Source evidence** is automatically retrieved from Azure AI Search — callers do not need to provide source chunks

**Two evaluation modes:**
- **Full-document mode** (`POST /api/v1/evaluate`) — evaluates the entire report as a single unit (original pipeline)
- **Section-wise mode** (`POST /api/v1/evaluate/sections`) — evaluates each section independently with section-specific retrieval, then aggregates scores

## Architecture

```
Client (M42 Pipeline)
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│   FastAPI (Azure Container App)                           │
│                                                           │
│   POST /api/v1/evaluate           (full-document mode)    │
│   POST /api/v1/evaluate/sections  (section-wise mode)     │
│                                                           │
│   ┌─────────────────────────────────────────────────────┐ │
│   │  Section-Wise Pipeline                              │ │
│   │                                                     │ │
│   │  1. Resolve input (JSON / Blob / PDF+DOCX)         │ │
│   │  2. Per-section retrieval from Azure AI Search      │ │     ┌────────────────────┐
│   │  3. Per-section LLM judge evaluation                │ │────►│ Azure AI Search    │
│   │  4. Aggregate section scores                        │ │     │ (source chunks)    │
│   └─────────────────────────────────────────────────────┘ │     └────────────────────┘
│                                                           │
│   ┌─────────────────────────────────────────────────────┐ │     ┌────────────────────┐
│   │  Full-Document Pipeline                             │ │────►│ Azure Cosmos DB    │
│   │                                                     │ │     │ (evaluations)      │
│   │  1. Retrieve chunks from Azure AI Search            │ │     └────────────────────┘
│   │  2. Build evaluation prompt with evidence           │ │
│   │  3. Run LLM judge                                   │ │     ┌────────────────────┐
│   └─────────────────────────────────────────────────────┘ │────►│ Azure Blob Storage │
│                                                           │     │ (reports + JSON)   │
│   ┌─────────────────────────────────────────────────────┐ │     └────────────────────┘
│   │  Azure OpenAI (GPT-4o) — LLM Judge                 │ │
│   │  + Enhanced Guidance Prompts + Chain-of-Thought     │ │     ┌────────────────────┐
│   └─────────────────────────────────────────────────────┘ │────►│ Azure Doc Intel    │
│                                                           │     │ (PDF/DOCX extract) │
└───────────────────────────────────────────────────────────┘     └────────────────────┘
```

## Azure Services Used

All services are **pre-provisioned**. The application does NOT create or provision any resources.

| Service | Purpose | Auth Method |
|---------|---------|-------------|
| Azure OpenAI (GPT-4o) | LLM judge for evaluation | Managed Identity (`DefaultAzureCredential`) |
| Azure Cosmos DB (NoSQL) | Store evaluation results | Managed Identity |
| Azure AI Search | Automatically retrieve source evidence chunks | Managed Identity |
| Azure Blob Storage | Store evaluation reports and structured JSON | Managed Identity |
| Azure Document Intelligence | Extract text from PDF/DOCX documents (section-wise mode) | Managed Identity |
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
| `BLOB_CONTAINER_NAME` | No | `evaluation-reports` | Blob container for evaluation reports |
| `BLOB_JSON_CONTAINER_NAME` | No | `cpg-report-json` | Blob container for structured report JSON |
| `DOCUMENT_INTELLIGENCE_ENDPOINT` | No* | — | `https://<resource>.cognitiveservices.azure.com/` |
| `USE_SECTION_MODE` | No | `true` | Enable section-wise evaluation pipeline |
| `LOG_LEVEL` | No | `INFO` | Logging level |

> \* `DOCUMENT_INTELLIGENCE_ENDPOINT` is required only if using section-wise mode with `file_path` input (PDF/DOCX extraction).

---

## API Endpoints

### `POST /api/v1/evaluate`

Evaluate a single CPG report across selected clinical dimensions (up to 8) using the full-document pipeline. Users can choose which metrics to evaluate via the `metrics` field.

### `POST /api/v1/evaluate/sections`

Evaluate a CPG report using section-wise evaluation. Accepts three input modes:
- **Inline JSON** (`report_json`) — pre-structured sections
- **Blob JSON path** (`json_path`) — load structured JSON from Azure Blob Storage
- **File path** (`file_path`) — extract from PDF/DOCX via Azure Document Intelligence

Each section is evaluated independently with section-specific retrieval, then scores are aggregated.

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
  "model": "gpt-4o",
  "metrics": ["clinical_accuracy", "safety_completeness", "hallucination_score", "fih_detected"],
  "reference_report": null,
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
| `model` | string | **Yes** | e.g., `"gpt-4o"`, `"gpt-5.1"`, `"gpt-4o-mini"` | Model used to generate the CPG report |
| `metrics` | list[string] | No | See table below. Min 1 required. Defaults to all 8 metrics | Which evaluation dimensions to run. Multi-select from frontend dropdown |
| `reference_report` | string | No | Any string or `null` | Gold-standard manually curated report for comparison (future use) |
| `evaluation_model` | string | No | `"gpt-4o"` (default), or any valid Azure OpenAI deployment name | LLM model used as judge |

### Available Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `clinical_accuracy` | Likert 1-5 | Are drug names, dosages, trial outcomes factually correct? |
| `completeness` | Likert 1-5 | Does the report cover all critical aspects from source evidence? |
| `safety_completeness` | Likert 1-5 | Are adverse effects, contraindications, dose modifications covered? |
| `relevance` | Likert 1-5 | Is all information directly pertinent to the guideline topic? |
| `coherence` | Likert 1-5 | Is the report logically structured and readable? |
| `evidence_traceability` | Likert 1-5 | Can every claim be traced back to a source chunk? |
| `hallucination_score` | Ordinal 1-4 | Does the report contain fabricated or contradictory claims? |
| `fih_detected` | List | Specific factually incorrect hallucinations with severity |

> **Note:** Source evidence chunks are automatically retrieved from Azure AI Search based on the `guideline_topic` and `disease_context`. You do not need to provide them in the request.
>
> **Note:** When a subset of metrics is selected, only those metrics appear in the response. Non-selected metrics are `null`.

---

## Output Format

### Successful Response — `200 OK`

```json
{
  "report_id": "rpt-20250319-ndmm-001",
  "evaluation_id": "eval-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-03-19T14:30:00.000Z",
  "generation_model": "gpt-4o",
  "evaluation_model": "gpt-4o",
  "num_runs": 1,
  "metrics_evaluated": ["clinical_accuracy", "completeness", "safety_completeness", "relevance", "coherence", "evidence_traceability", "hallucination_score", "fih_detected"],

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

  "confidence_level": "high",
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
| `generation_model` | string | e.g., `"gpt-4o"`, `"gpt-5.1"` | Model used to generate the CPG report |
| `evaluation_model` | string | `"gpt-4o"` or deployment name | LLM judge model used for evaluation |
| `num_runs` | int | Always `1` | Number of evaluation runs performed |
| `metrics_evaluated` | list[str] | Subset of 8 metric names | Which metrics were evaluated in this run |
| `clinical_accuracy` | object or null | See below, or `null` if not selected | Clinical accuracy metric result |
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
| `confidence_level` | string | `"high"` | Always high (single run) |
| `flags` | list[str] | See below | Critical issues detected |

### Possible Flag Values

| Flag | Trigger Condition |
|------|-------------------|
| `missing_safety_data` | `safety_completeness.score <= 2` |
| `safety_gaps_identified` | `safety_completeness.missing_items` is non-empty |
| `poor_evidence_traceability` | `evidence_traceability.score <= 2` |
| `untraced_claims_present` | `evidence_traceability.untraced_claims` is non-empty |
| `hallucinations_detected` | `hallucination_score.score <= 2` |
| `critical_fih_detected` | Any FIH with `severity == "critical"` |
| `fih_present` | Any FIH detected |
| `low_clinical_accuracy` | `clinical_accuracy.score <= 2` |

### Error Responses

| Status | Body | Condition |
|--------|------|-----------|
| `422` | Validation error details | Missing required fields or empty report |
| `500` | `{"detail": "Evaluation failed: <reason>"}` | Azure service error or no chunks retrieved |
| `503` | `{"detail": "Azure OpenAI service is throttled. Please retry later."}` | Rate limiting |

---

## Section-Wise Evaluation

### `POST /api/v1/evaluate/sections`

### Input Payload

The section-wise endpoint accepts three input modes. Provide exactly one of `report_json`, `json_path`, or `file_path`.

#### Mode A — Inline JSON (preferred)

```json
{
  "guideline_topic": "First-line treatment for transplant-eligible NDMM",
  "disease_context": "Multiple Myeloma",
  "metrics": ["clinical_accuracy", "safety_completeness", "hallucination_score", "fih_detected"],
  "evaluation_model": "gpt-4o",
  "report_json": {
    "report_id": "rpt-001",
    "sections": [
      {
        "id": "sec-1",
        "title": "Introduction",
        "content": "This guideline covers first-line treatment options for NDMM...",
        "section_type": "general",
        "order": 0,
        "keywords": ["NDMM", "treatment", "first-line"]
      },
      {
        "id": "sec-2",
        "title": "Treatment Recommendations",
        "content": "D-VRd is recommended as first-line therapy...",
        "section_type": "guideline",
        "order": 1,
        "keywords": ["D-VRd", "GRIFFIN", "therapy"]
      }
    ]
  }
}
```

#### Mode A — Blob JSON Path

```json
{
  "guideline_topic": "First-line treatment for transplant-eligible NDMM",
  "disease_context": "Multiple Myeloma",
  "json_path": "cpg-report-json/rpt-001.json"
}
```

#### Mode B — Raw Document (PDF/DOCX)

```json
{
  "guideline_topic": "First-line treatment for transplant-eligible NDMM",
  "disease_context": "Multiple Myeloma",
  "file_path": "https://<account>.blob.core.windows.net/documents/report.pdf"
}
```

When using `file_path`, the document is processed via Azure Document Intelligence (`prebuilt-layout`) to extract paragraphs, headings, and tables. Sections are automatically detected using heading heuristics (DI role metadata, numbered headers, ALL CAPS lines, short title lines) with a fallback to paragraph chunking.

### Section JSON Schema

```json
{
  "report_id": "string",
  "sections": [
    {
      "id": "string",
      "title": "string",
      "content": "string",
      "section_type": "string",
      "order": 0,
      "keywords": ["string"]
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sections[].id` | string | Unique section identifier (UUID) |
| `sections[].title` | string | Section heading |
| `sections[].content` | string | Section text content |
| `sections[].section_type` | string | Inferred type: `definitions`, `abbreviations`, `guideline`, `general` |
| `sections[].order` | int | Position in the document |
| `sections[].keywords` | list[str] | Top 5-10 keywords extracted via TF-IDF |

### Section-Wise Output

```json
{
  "report_id": "rpt-001",
  "evaluation_id": "eval-uuid",
  "timestamp": "2025-03-24T14:30:00Z",
  "evaluation_model": "gpt-4o",
  "metrics_evaluated": ["clinical_accuracy", "safety_completeness", "hallucination_score", "fih_detected"],

  "final_scores": {
    "clinical_accuracy": 3.5,
    "safety_completeness": 2.5,
    "hallucination_score": 3.0
  },

  "section_scores": [
    {
      "section_id": "sec-1",
      "section_title": "Introduction",
      "section_type": "general",
      "clinical_accuracy": { "score": 4, "confidence": "high", "reasoning": "..." },
      "safety_completeness": { "score": 3, "confidence": "medium", "reasoning": "...", "missing_items": [] },
      "hallucination_score": { "score": 3, "confidence": "high", "reasoning": "..." },
      "fih_detected": [],
      "flags": []
    },
    {
      "section_id": "sec-2",
      "section_title": "Treatment Recommendations",
      "section_type": "guideline",
      "clinical_accuracy": { "score": 3, "confidence": "medium", "reasoning": "..." },
      "safety_completeness": { "score": 2, "confidence": "high", "reasoning": "...", "missing_items": ["AE rates"] },
      "hallucination_score": { "score": 3, "confidence": "high", "reasoning": "..." },
      "fih_detected": [],
      "flags": ["missing_safety_data", "safety_gaps_identified"]
    }
  ],

  "confidence_level": "high",
  "flags": ["missing_safety_data", "safety_gaps_identified"],
  "cosmos_document_id": "eval-uuid",
  "blob_url": "https://..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `final_scores` | dict[str, float] | Aggregated average scores across all sections (excludes `fih_detected`) |
| `section_scores` | list[SectionScore] | Per-section evaluation results with individual metrics and flags |
| `section_scores[].flags` | list[str] | Flags specific to that section |
| `flags` | list[str] | Deduplicated union of all section flags |

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
DOC_INTEL_RESOURCE_NAME="your-doc-intel-resource"
```

### Step 2 — Build and push the container image to ACR

Build the Docker image locally and push it to Azure Container Registry:

```bash
cd m42-evaluation-api
docker build -t $IMAGE_TAG .
az acr login --name $ACR_NAME
docker push $IMAGE_TAG
```

### Step 3 — Enable ACR admin access and retrieve credentials

```bash
az acr update --name $ACR_NAME --admin-enabled true
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
```

### Step 4 — Create the Container App

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
    BLOB_ACCOUNT_URL="https://$STORAGE_ACCOUNT_NAME.blob.core.windows.net" \
    BLOB_CONTAINER_NAME="evaluation-reports" \
    BLOB_JSON_CONTAINER_NAME="cpg-report-json" \
    DOCUMENT_INTELLIGENCE_ENDPOINT="https://$DOC_INTEL_RESOURCE_NAME.cognitiveservices.azure.com/" \
    USE_SECTION_MODE="true" \
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

**Azure Document Intelligence — Cognitive Services User** (only if using section-wise mode with PDF/DOCX):

```bash
DOC_INTEL_RESOURCE_ID=$(az cognitiveservices account show \
  --name $DOC_INTEL_RESOURCE_NAME --resource-group $RESOURCE_GROUP \
  --query id -o tsv)

az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services User" \
  --scope $DOC_INTEL_RESOURCE_ID
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
| Azure Document Intelligence | Cognitive Services User | Document Intelligence resource |
