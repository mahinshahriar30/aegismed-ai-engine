# HouseMD 

AI medical engine specialized in high-precision clinical diagnosis, emergency triage, and deterministic RAG execution.

## Overview

HouseMD is an enterprise-grade, deterministic Retrieval-Augmented Generation (RAG) microservice built with FastAPI, ChromaDB, and the Google GenAI SDK. Designed for high-stress healthcare environments—such as emergency departments, ICU corridor-clearing systems, and rural diagnostic centers—the engine automates the evaluation of unstructured medical presentation notes, emergency triage reports, and clinical lab findings.

To eliminate the key operational risks of clinical AI, HouseMD incorporates three core design principles:

- **Deterministic Vector Grounding:** Cross-references incoming clinical cases against an indexed ChromaDB knowledge base of emergency protocols (e.g., FAST stroke criteria, cardiac troponin thresholds, and DUMBBELSS toxicology profiles).
- **10-Model Resilient Failover Cascade:** Automatically switches across a 10-model fallback sequence (`gemini-2.5-flash`, `gemini-3.6-flash`, `gemini-2.5-pro`, etc.) to survive rate limits and API outages seamlessly.
- **Strict Pydantic Contracts:** Guarantees 100% structured JSON output (`AegisMedAuditResponse`), eliminating JSON parsing errors for downstream Electronic Health Record (EHR) systems and frontend dashboards.

## Quick Start

```bash
git clone [https://github.com/mahinshahriar30/aegismed-ai-engine.git](https://github.com/mahinshahriar30/aegismed-ai-engine.git)
cd aegismed-ai-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
nohup uvicorn src.api:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

```

Access API documentation at `http://YOUR_IP:8000/docs`

## Features

* **Zero-Hallucination Grounded Triage:** Grounded RAG reasoning cited against indexed clinical guidelines.
* **10-Model Resilient Failover:** Automated, sub-second failover loop across 10 Google Gemini models.
* **Domain-Aware Guideline Chunking:** Custom splitting on `[DIAGNOSTIC_REF:` boundary tags to preserve complete diagnostic blocks.
* **Strict Type Enforcement:** Native GenAI schema binding with Pydantic for zero-error JSON responses.
* **Isolated Testing Pipeline:** Modular unit testing via `test_engine.py` without server overhead or DB mutation.

## API

### GET /health

Returns `{"status": "ok", "system": "HouseMD Engine Operational"}`

### POST /api/v1/diagnose

Request:

```json
{
  "document_text": "PATIENT PRESENTATION: 54-year-old male with central crushing chest pain radiating to left jaw, diaphoresis. Vitals: BP 140/90, HR 105. Labs: Cardiac Troponin I: 0.85 ng/mL.",
  "domain": "medical"
}

```

Response:

```json
{
  "patient_summary": "54-year-old male presenting with central crushing chest pain, diaphoresis, and elevated Cardiac Troponin I (0.85 ng/mL).",
  "primary_diagnosis": {
    "condition_name": "Acute Myocardial Infarction (AMI)",
    "triage_level": "CRITICAL_EMERGENCY",
    "clinical_justification": "Cardiac Troponin I level of 0.85 ng/mL significantly exceeds the diagnostic threshold (0.04 ng/mL) accompanied by typical ischemic symptoms.",
    "reference_guideline": "ACUTE_CORONARY_SYNDROME_AND_MI"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Obtain immediate 12-lead ECG within 10 minutes",
    "Administer antiplatelet therapy as per acute ACS protocol",
    "Activate Emergency Cardiac Cath Lab for urgent PCI referral"
  ]
}

```

## Testing

```bash
python -m test_engine

```

## Configuration

Set your environment variables in your active shell session or `.env` file:

```bash
GEMINI_API_KEY="your_actual_gemini_api_key_here"
PORT=8000

```

## Project Structure

```
housemd/
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml      # CI/CD Hugging Face deployment pipeline
├── data/
│   └── medical_reference.txt   # Indexed clinical guidelines & triage rules
├── src/
│   ├── api.py                  # FastAPI routes, lifecycle management & validation
│   ├── database.py             # ChromaDB vector store & SentenceTransformers embeddings
│   ├── engine.py               # RAG prompt construction & 10-model fallback cascade
│   └── schema.py               # Pydantic models & triage enumerations
├── Dockerfile                  # Container definition for HF Spaces & cloud hosting
├── requirements.txt            # Python dependencies
└── test_engine.py              # Standalone diagnostic pipeline test

```

## Management

```bash
# Start (foreground)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Start (background)
nohup uvicorn src.api:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Stop
lsof -ti:8000 | xargs kill -9

# Logs
tail -f server.log

```

## Troubleshooting

**Port in use:** `lsof -ti:8000 | xargs kill -9`

**Import errors:** `export PYTHONPATH=$(pwd):$PYTHONPATH`

**Missing API Key:** Verify that `GEMINI_API_KEY` is correctly exported in your environment.

## License

Copyright (c) 2026 HouseMD Team. All rights reserved.

```

```