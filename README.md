# 🏥 HouseMD AI Engine (AegisMed)

### *Guideline-Grounded Emergency Clinical Triage Framework*

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-Netlify_CDN-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://housemdai.netlify.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker_Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logoColor=white)](https://groq.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

---

**A guideline-grounded Clinical Decision Support System (CDSS) prototype for Emergency Departments, Rural Diagnostic Centers, and Mass Casualty Incidents.**
👉 **Live Application:** [housemdai.netlify.app](https://housemdai.netlify.app)

[Overview](#-overview) • [Key Features](#-key-features) • [How It Works](#-how-it-works) • [Guidelines](#-indexed-clinical-guidelines-36-conditions) • [Quick Start](#-quick-start) • [Configuration](#-configuration) • [API](#-api-documentation) • [Limitations](#-limitations) • [License](#-license)

---

## 💡 Overview

In emergency triage, an AI model that invents a diagnosis is worse than one that says "I'm not sure". **AegisMed** is built around that idea: the diagnosis and triage level shown to the clinician come from a **curated guideline database**, not from free-form model output.

1. **Search:** The presentation is matched against 36 emergency guidelines stored in ChromaDB, using a vector search plus a keyword check.
2. **Grounded AI:** If a guideline matches, only the matched guideline text is given to the LLM, which writes the summary, justification and action steps. The condition name and triage level in the response are taken from the database entry, and the reference guideline is attached by code, not by the model.
3. **Database fallback:** If every AI model fails, the API still answers with the matched guideline's condition name, triage level and treatment steps.
4. **Not Sure:** If nothing matches, the API answers **"Not Sure" / `UNDETERMINED`** and asks for manual triage. The model is never asked to diagnose freely.

The API is designed to always return a valid answer: provider outages, malformed model output and database errors all end in one of the modes above, not a server error.

---

## ✨ Key Features

- 🎯 **Guideline-grounded answers:** Headline diagnosis and triage level come from the curated database; the LLM only explains and elaborates.
- 🔎 **Hybrid retrieval:** Vector search (ChromaDB) combined with a keyword check over each guideline's diagnostic criteria, because short symptom notes match long protocol text poorly with embeddings alone.
- 🛟 **Graceful degradation:** AI models down → database result. No match → "Not Sure". Never a crash.
- 🧾 **Traceable references:** The guideline titles shown to the user are added by code from what was actually retrieved.
- 🛡️ **Multi-model failover:** Tries `openai/gpt-oss-20b`, `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on Groq in order; a model only counts as successful if its output parses and validates. Gemini models can be added to the cascade.
- 📋 **Strict typing:** Pydantic schema (`AegisMedAuditResponse`) with a `CRITICAL_EMERGENCY` / `HIGH_PRIORITY` / `STABLE` / `UNDETERMINED` triage enum.
- 🔒 **Secured API:** `X-API-Key` header (constant-time comparison, no default key), restricted CORS origin, request length limit.
- 📦 **Self-loading knowledge base:** The guideline database is rebuilt from `data/medical_reference.txt` automatically at startup when it is missing or out of date.
- 🌐 **Decoupled deployment:** Static frontend on Netlify, containerized FastAPI backend on Render, keep-alive heartbeat for free-tier hosting.

---

## 🏗️ How It Works

```
 Clinical note ──► POST /api/v1/diagnose ──► Hybrid search in ChromaDB
                                              (vector distance + keyword check)
                                                          │
                          ┌───────────────────────────────┴───────────────────────────┐
                          ▼                                                           ▼
                 Matching guideline(s) found                                 No guideline matches
                          │                                                           │
                          ▼                                                           ▼
            Matched guideline text ──► LLM cascade                        "Not Sure" / UNDETERMINED
                          │                                               (manual triage required)
              ┌───────────┴───────────┐
              ▼                       ▼
        AI succeeds             All AI models fail
              │                       │
              ▼                       ▼
   source: ai_grounded        source: database_fallback
   Condition + triage from    Condition + triage + steps read
   the database; LLM adds     directly from the top matching
   summary, justification,    guideline; no AI reasoning
   actions                    applied (clearly flagged)
```

### What counts as a "match"

A guideline is a match if **either** condition holds:

- its vector distance is at most `MAX_MATCH_DISTANCE` (default `0.9`), **or**
- at least `LEX_MIN_TERMS` (default `3`) distinctive words from the note appear in the guideline's criteria, with a weighted score of at least `LEX_MIN_SCORE` (default `7.0`).

Matched guidelines are ranked first, then ordered by combining the distance rank with the keyword rank. Only matched guidelines are ever sent to the LLM.

### Response modes (`source` field)

| `source` | Meaning |
|---|---|
| `ai_grounded` | A guideline matched and the LLM produced the explanation. Condition and triage come from the database. |
| `database_fallback` | A guideline matched but every AI model failed. Result read directly from the database. |
| `no_match` | No guideline matched (or the knowledge base was unavailable). Condition is "Not Sure", triage is `UNDETERMINED`. |

---

## 📚 Indexed Clinical Guidelines (36 Conditions)

> 🩺 **Medical Review Note:** The 36 reference protocols in `data/medical_reference.txt` are curated for standard acute care, resuscitation and toxicology practice. Each entry starts with `[DIAGNOSTIC_REF: NAME]` and lists diagnostic criteria, primary diagnosis, urgency and a numbered treatment protocol.

| Category | Indexed Emergency Conditions |
| --- | --- |
| **Cardiovascular & Cerebrovascular** | Acute Ischemic/Hemorrhagic Stroke, Acute Coronary Syndrome (STEMI/NSTEMI), Acute Pulmonary Embolism, Hypertensive Emergency, Acute Decompensated Heart Failure |
| **Trauma & Resuscitation** | Traumatic Hemorrhagic Shock, Polytrauma / Road Traffic Accident, Traumatic Brain Injury (TBI/EDH), Tension Pneumothorax, Cardiac Arrest (ACLS), Drowning Asphyxia |
| **Toxico-Environmental** | Organophosphate Toxicity, Paracetamol Overdose, Benzodiazepine Overdose, Aluminum Phosphide (Rice Tablet), Methanol Poisoning, Carbon Monoxide Poisoning, Heat Stroke, Snakebite Envenomation |
| **Sepsis & Infectious Disasters** | Severe Septic Shock, Dengue Hemorrhagic Fever (DHF), Severe Falciparum Malaria, Acute Watery Diarrhea / Cholera Shock, Acute Bacterial Meningitis, Rabies Exposure |
| **Obstetrics & Metabolic** | Eclampsia / Severe Preeclampsia, Diabetic Ketoacidosis (DKA/HHS), Status Epilepticus, Acute Kidney Injury (AKI), Acute Pancreatitis, Upper GI Bleeding, Perforated Peptic Ulcer, Acute Liver Failure, Severe Acute Malnutrition (SAM) |

To add or edit a guideline, edit `data/medical_reference.txt` (keep the `[DIAGNOSTIC_REF: NAME]` header and the `- Urgency:`, `- Primary Diagnosis:` and `- Diagnostic Criteria:` lines) and restart the service. The database is rebuilt automatically.

---

## 🚀 Quick Start

### 1. Clone & set up the environment

```bash
git clone https://github.com/mahinshahriar30/aegismed-ai-engine.git
cd aegismed-ai-engine

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

python -m pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root (never commit it):

```
AEGIS_API_KEY="choose_a_long_random_secret"
GROQ_API_KEY="your_groq_api_key"
```

See [Configuration](#-configuration) for all options.

### 3. Build the knowledge base and check retrieval

```bash
# Build the ChromaDB collection from data/medical_reference.txt (should report 36 guidelines)
python -m src.database --ingest

# See how a presentation is matched (distance, keyword score, MATCH / no match)
python -m src.database "62-year-old male with facial drooping, slurred speech, right arm weakness, BP 195/110"
```

The app also does this automatically at startup, so this step is mainly for local checks and threshold tuning.

### 4. Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 10000 --env-file .env
```

Interactive API docs (Swagger UI): `http://localhost:10000/docs`

### 5. Or run with Docker

```bash
docker build -t aegismed-ai-engine .
docker run -p 10000:10000 --env-file .env aegismed-ai-engine
```

> The first start downloads the embedding model (about 80 MB) and loads the guidelines. Until loading finishes, requests are answered "Not Sure"; `/health` shows `guideline_chunks` once it is ready.

---

## ⚙️ Configuration

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AEGIS_API_KEY` | **Yes** | none | Secret checked against the `X-API-Key` header. If unset, every diagnose request is rejected. |
| `GROQ_API_KEY` | Yes, for AI answers | none | Enables the Groq models. Without it, the API still works in `database_fallback` / `no_match` mode. |
| `GEMINI_API_KEY` | No | none | Only used if Gemini model names are added to `CASCADE_MODELS` in `src/engine.py`. |
| `ALLOWED_ORIGINS` | No | `https://housemdai.netlify.app` | Comma-separated list of frontend origins allowed by CORS. |
| `MAX_MATCH_DISTANCE` | No | `0.9` | Vector-distance cutoff for a match (smaller is stricter). |
| `LEX_MIN_TERMS` | No | `3` | Minimum distinctive keywords needed for a keyword match. |
| `LEX_MIN_SCORE` | No | `7.0` | Minimum weighted keyword score for a keyword match. |
| `LLM_TIMEOUT_SECONDS` | No | `25` | Per-model timeout. |
| `CHROMA_DB_PATH` | No | `./chroma_db` | Where the vector database is stored. |
| `MEDICAL_DATA_FILE` | No | `data/medical_reference.txt` | Guideline source file. |
| `PORT` | No | `10000` | Port used by the container (Render sets this). |
| `RENDER_EXTERNAL_URL` | No | set by Render | Enables the keep-alive heartbeat when present. |

---

## 📡 API Documentation

### 1. Health Check & Heartbeat (unauthenticated)

```
GET /health
```

```json
{
  "status": "healthy",
  "service": "AegisMed AI Engine",
  "version": "2.1.0",
  "rag_status": "initialized",
  "guideline_chunks": 36
}
```

`rag_status` is `initialized`, `empty` (still loading or no data) or `unavailable` (database error).

### 2. Clinical Diagnosis Audit

```
POST /api/v1/diagnose
Headers:
  X-API-Key: your_secret_key_here
  Content-Type: application/json
```

Request body (`document_text`: 1 to 8000 characters):

```json
{
  "document_text": "EMERGENCY PRESENTATION: 28-year-old agricultural worker presenting with sudden pinpoint pupils (miosis), profuse salivation, vomiting, wheezing, and bradycardia (HR 48 bpm, BP 85/55 mmHg) after crop spraying."
}
```

The endpoint returns HTTP 200 with a valid audit in all three modes below.

**Mode 1: `ai_grounded`**

```json
{
  "patient_summary": "28-year-old male presenting with cholinergic crisis symptoms following agricultural pesticide exposure.",
  "primary_diagnosis": {
    "condition_name": "Acute Organophosphate Insecticide Toxicity",
    "triage_level": "CRITICAL_EMERGENCY",
    "clinical_justification": "Presentation matches the cholinergic crisis criteria (miosis, salivation, bronchospasm, bradycardia, hypotension).",
    "reference_guideline": "ORGANOPHOSPHATE_POISONING_TOXICOLOGY"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Remove contaminated clothing and wash skin with soap and cold water (staff PPE required)",
    "Administer Atropine 2 mg to 5 mg IV every 5-10 minutes until full atropinization",
    "Administer Pralidoxime (2-PAM) 1 g to 2 g IV over 15-30 minutes, then continuous infusion",
    "Prepare for early endotracheal intubation if respiratory failure persists"
  ],
  "source": "ai_grounded",
  "grounded": true,
  "reference_guidelines": ["ORGANOPHOSPHATE_POISONING_TOXICOLOGY"],
  "notice": null
}
```

**Mode 2: `database_fallback`** (all AI models unavailable; steps abbreviated here)

```json
{
  "patient_summary": "AI analysis is temporarily unavailable. The closest matching guideline in the local database is 'ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE'.",
  "primary_diagnosis": {
    "condition_name": "Acute Cerebrovascular Accident (Acute Ischemic Stroke or Hemorrhagic Stroke)",
    "triage_level": "CRITICAL_EMERGENCY",
    "clinical_justification": "Matched to a stored guideline by text similarity. No AI reasoning was applied.",
    "reference_guideline": "ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Airway & Oxygenation: Maintain SpO2 > 94%; establish IV access; perform immediate blood glucose check to rule out hypoglycemia.",
    "Urgent Imaging: Order non-contrast head CT scan immediately to differentiate ischemic vs. hemorrhagic stroke."
  ],
  "source": "database_fallback",
  "grounded": true,
  "reference_guidelines": ["ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE"],
  "notice": "AI models are unavailable. This result comes directly from the closest matching guideline in the local database, with no AI reasoning applied. Verify clinically."
}
```

**Mode 3: `no_match`**

```json
{
  "patient_summary": "No matching guideline was found for this presentation.",
  "primary_diagnosis": {
    "condition_name": "Not Sure",
    "triage_level": "UNDETERMINED",
    "clinical_justification": "No verified guideline could be matched and no AI diagnosis was generated.",
    "reference_guideline": "None"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Perform immediate manual clinical triage and assessment.",
    "Consult local emergency guidelines and a senior clinician."
  ],
  "source": "no_match",
  "grounded": false,
  "reference_guidelines": [],
  "notice": "No matching guideline was found in the local knowledge base, and no AI diagnosis was generated. Manual clinical triage is required."
}
```

Errors: `401` for a missing or wrong API key, `422` for an empty or over-long `document_text`.

---

## 🌍 Deployment

**Backend (Render, Docker):**
1. Create a Web Service from this repository (Docker runtime).
2. Set `AEGIS_API_KEY` and `GROQ_API_KEY` under **Environment**. Optionally set `ALLOWED_ORIGINS`.
3. Deploy, then open `/health` and confirm `guideline_chunks` is `36`.

**Frontend (Netlify):** Deploy `src/index.html`. It needs `BACKEND_URL` and the same API key as `AEGIS_API_KEY`.

> **Note on the frontend key:** A key embedded in a static page is visible to anyone who opens the page source, so treat it as an identifier, not a secret. Use rate limiting, the CORS origin restriction and the input length limit as the real protection, or route requests through a server-side proxy (for example a Netlify function) that holds the key.

---

## 📂 Project Structure

```
aegismed-ai-engine/
├── data/
│   └── medical_reference.txt  # 36 clinical emergency guidelines (one [DIAGNOSTIC_REF] block each)
├── src/
│   ├── api.py                 # FastAPI service: auth, CORS, startup loading, keep-alive loop
│   ├── database.py            # ChromaDB access, guideline loader, hybrid (vector + keyword) retrieval
│   ├── engine.py              # Grounded prompting, LLM failover cascade, database fallback, "Not Sure"
│   ├── index.html             # Single-page frontend (Netlify)
│   └── schema.py              # Pydantic models and triage enum
├── Dockerfile                 # Non-root Docker build for Render
├── netlify.toml               # Netlify deployment config
├── requirements.txt           # Python dependencies
└── test_engine.py             # End-to-end test of the sample cases (run: python test_engine.py)
```

---

## 🧪 Tuning & Testing

Use the retrieval inspector to check how any note is matched:

```bash
python -m src.database "your clinical note here"
```

It prints the five best candidates with their vector distance, keyword score/terms and whether each counts as a `MATCH`. Run `python test_engine.py` to check the sample cases end to end. Without `GROQ_API_KEY` set it exercises the no-AI path (matched cases return `database_fallback`); with the key set it exercises the AI path (`ai_grounded`). The unrelated case should always return `no_match`.

The default thresholds were calibrated on the five sample cases in the web app plus a few non-matching notes. If you add guidelines or notice missed matches, inspect a few real notes with the command above and adjust `MAX_MATCH_DISTANCE`, `LEX_MIN_TERMS` and `LEX_MIN_SCORE`.

---

## 🚧 Limitations

- **Prototype only.** Retrieval thresholds were tuned on a small set of examples; they have not been validated on real clinical notes.
- **Keyword matching is lexical.** It matches words, not meaning, so unusual phrasing can miss a guideline (the safe result is "Not Sure") and a note that repeats a guideline's own words can match it without being that condition.
- **The embedding model is general-purpose** (MiniLM via ChromaDB's default), not clinical, which is why the keyword check exists.
- **The database fallback shows the single closest guideline** and is clearly flagged; a clinician must verify it.
- **Only the knowledge base is covered:** conditions outside the 36 guidelines return "Not Sure".
- **Free-tier hosting** can have cold starts, and Render's disk is ephemeral, which is why the database is rebuilt at startup.

---

## ⚠️ Research Disclaimer

> **RESEARCH PROTOTYPE ONLY:** AegisMed AI Engine (HouseMD) is an experimental software proof-of-concept created strictly for technical demonstration, software benchmarking, and academic research evaluation. It is **NOT** a certified medical device and must **NEVER** be used for active patient care, diagnosis, or triage during real clinical emergencies. Do not enter real patient data.

---

## 📜 License

Copyright (c) 2026 Mahin Shahriar. All rights reserved.