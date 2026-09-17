
---

```markdown
# 🏥 HouseMD AI Engine (AegisMed)

> **Deterministic Dual-Tier Emergency Clinical Decision Support & Mass Casualty Triage Engine**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6F00?logo=chromadb)](https://www.trychroma.com/)
[![Google GenAI SDK](https://img.shields.io/badge/LLM-Google_Gemini-4285F4?logo=google)](https://ai.google.dev/)
[![Streamlit Cloud](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Deployment-Render-46E3B7?logo=render)](https://render.com/)

---

## 📌 Overview

**HouseMD AI Engine** (also known as **AegisMed**) is an enterprise-grade microservice built with **FastAPI**, **ChromaDB**, and the **Google GenAI SDK**, coupled with a **Streamlit** frontend interface. Designed for high-stress healthcare environments—such as emergency departments, mass casualty incidents (MCI), ICU corridor-clearing systems, and rural triage centers—the engine automates the evaluation of unstructured medical presentation notes, emergency reports, and lab results.

To eliminate clinical AI risks while ensuring high availability, HouseMD operates on three core principles:

- **Dual-Tier Safety Protocol:** Enforces deterministic, **zero-hallucination vector grounding** against 36 indexed emergency guidelines for critical cases, while enabling a safe **foundation model fallback** for unindexed or stable presentations.
- **Resilient Model Failover Cascade:** Automatically cycles through a fallback sequence of Google Gemini models to handle rate limits and transient outages seamlessly.
- **Strict Pydantic Type Enforcement:** Guarantees 100% structured JSON outputs (`AegisMedAuditResponse`) for direct integration with hospital Electronic Health Record (EHR) systems and frontend dashboards.

---

## ⚡ System Architecture: Dual-Tier Triage Framework

During mass casualties or clinical emergencies, healthcare providers require immediate, non-hallucinated intervention steps while maintaining a safe screening process for lower-priority presentations.


```

```
                          [ Incoming Patient Presentation ]
                                          │
                                          ▼
                             [ ChromaDB Vector Query ]
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              ▼                                                       ▼
    [ Match Found in Index ]                               [ No Index Match Found ]

```

(36 Indexed Emergency Guidelines)                           (Unindexed / Non-Critical)
│                                                       │
▼                                                       ▼
┌───────────────────────────┐                           ┌───────────────────────────┐
│ TIER 1: STRICT GROUNDING  │                           │ TIER 2: GENERAL FALLBACK  │
│ • Zero Hallucination      │                           │ • Foundation Model Logic  │
│ • Grounded in Protocol    │                           │ • Lower Triage Priority   │
│ • Step-by-Step Action Plan│                           │ • Standard Clinical Review│
└───────────────────────────┘                           └───────────────────────────┘

```

1. **Tier 1 — Strict Verified Guideline Matching (`CRITICAL_EMERGENCY` / `HIGH_PRIORITY`):**
   * Cross-references incoming clinical data against **36 pre-indexed emergency guidelines** stored in ChromaDB.
   * Forces Gemini to base diagnoses, triage levels, and step-by-step emergency actions exclusively on verified reference guidelines.
2. **Tier 2 — General Emergency Reasoning Fallback (`STABLE` / Unindexed):**
   * For non-indexed presentations, the engine gracefully leverages Gemini’s medical foundation knowledge to evaluate the patient while explicitly tagging the output as an unindexed general clinical assessment.

---

## 📋 Indexed Clinical Guidelines Scope (36 Conditions)

The internal vector repository (`medical_reference.txt`) indexes 36 high-yield emergency conditions complete with diagnostic triggers and step-by-step immediate treatment protocols:

* **Cardiovascular & Cerebrovascular:** Acute Stroke (CVA), Acute Coronary Syndrome & MI (STEMI/NSTEMI), Acute Pulmonary Embolism (PE), Hypertensive Emergency, Acute Decompensated Heart Failure & Pulmonary Edema.
* **Trauma & Resuscitation:** Severe Traumatic Hemorrhagic Shock, Polytrauma/RTA, Traumatic Brain Injury (TBI/EDH), Tension Pneumothorax, Sudden Cardiac Arrest (ACLS), Drowning & Near-Drowning Asphyxia.
* **Toxico-Environmental:** Organophosphate Toxicity, Paracetamol Overdose, Benzodiazepine Toxicity, Aluminum Phosphide (Rice Tablet) Poisoning, Methanol Poisoning, Carbon Monoxide Poisoning, Heat Stroke, Snakebite Envenomation.
* **Sepsis & Infectious Disasters:** Severe Sepsis & Septic Shock, Dengue Hemorrhagic Fever (DHF), Severe Falciparum Malaria, Severe Acute Watery Diarrhea & Cholera Shock, Acute Bacterial Meningitis, Rabies Lyssavirus Exposure.
* **Obstetrics, Gastrointestinal & Metabolic:** Eclampsia & Severe Preeclampsia, Diabetic Ketoacidosis (DKA/HHS), Status Epilepticus, Acute Kidney Injury (AKI), Acute Pancreatitis, Acute Upper GI Bleeding, Perforated Peptic Ulcer & Peritonitis, Fulminant Hepatic Encephalopathy, Severe Acute Malnutrition (SAM).

---

## 🚀 Quick Start & Local Management

### 1. Installation

```bash
git clone [https://github.com/mahinshahriar30/aegismed-ai-engine.git](https://github.com/mahinshahriar30/aegismed-ai-engine.git)
cd aegismed-ai-engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```

### 2. Set Environment Variables

```bash
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
export PORT=8000

```

### 3. Run Application

```bash
# Start FastAPI backend (Foreground)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Start FastAPI backend (Background)
nohup uvicorn src.api:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Start Streamlit UI
streamlit run app.py

```

Access Interactive API Documentation at `http://localhost:8000/docs`

---

## 📡 API Endpoint Reference

### `GET /health`

Returns system status.

```json
{"status": "ok", "system": "HouseMD Engine Operational"}

```

### `POST /api/v1/diagnose`

**Request:**

```json
{
  "document_text": "EMERGENCY ROOM ASSESSMENT: 28-year-old agricultural worker with accidental pesticide exposure. Pinpoint pupils (miosis), profuse salivation, vomiting, wheezing. HR 48 bpm, BP 85/55 mmHg.",
  "domain": "medical"
}

```

**Response:**

```json
{
  "patient_summary": "28-year-old agricultural worker presenting with cholinergic crisis symptoms following pesticide exposure, including miosis, profuse salivation, wheezing, bradycardia (HR 48 bpm), and hypotension (BP 85/55 mmHg).",
  "primary_diagnosis": {
    "condition_name": "Acute Organophosphate Insecticide Toxicity",
    "triage_level": "CRITICAL_EMERGENCY",
    "clinical_justification": "Presentation matches classic cholinergic crisis (DUMBBELSS criteria: miosis, salivation, bronchospasm, bradycardia, hypotension) following pesticide exposure.",
    "reference_guideline": "ORGANOPHOSPHATE_POISONING_TOXICOLOGY"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Remove contaminated clothing and wash skin with soap and cold water immediately (PPE required for staff)",
    "Administer Atropine 2 mg to 5 mg IV bolus every 5-10 minutes until full atropinization (clearing of lungs, HR > 80 bpm)",
    "Administer Pralidoxime (2-PAM) 1 g to 2 g IV over 15-30 minutes followed by continuous infusion",
    "Prepare for early endotracheal intubation if respiratory failure or excessive bronchorrhea persists"
  ]
}

```

---

## 📂 Project Structure

```
housemd/
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml      # CI/CD deployment pipeline
├── data/
│   └── medical_reference.txt   # 36 indexed clinical reference guidelines & protocols
├── src/
│   ├── api.py                  # FastAPI routes, lifecycle management & middleware
│   ├── database.py             # ChromaDB vector store initialization & querying
│   ├── engine.py               # Dual-tier RAG prompt construction & model failover cascade
│   └── schema.py               # Pydantic data contracts & triage enumerations
├── app.py                      # Streamlit frontend web application
├── Dockerfile                  # Container definition for Render / cloud deployment
├── requirements.txt            # Python dependencies
└── test_engine.py              # Modular test suite for diagnostic pipeline

```

---

## 🧪 Testing

Run the isolated testing pipeline without modifying persistent database state:

```bash
python -m test_engine

```

---

## 🌐 Cloud Deployment Architecture

* **Backend Web Service:** FastAPI + ChromaDB deployed on **Render** (via custom Dockerfile).
* **Frontend Web Application:** Streamlit UI deployed on **Streamlit Community Cloud**.
* **Inter-Service Communication:** Streamlit connects to Render via the `BACKEND_URL` secret.

---

## 🚨 Disclaimer & Research Notice

> **RESEARCH PROTOTYPE ONLY:** HouseMD AI Engine (AegisMed) is an experimental software proof-of-concept created strictly for software benchmarking, research evaluation, and technical demonstrations. It is **NOT** a certified medical device and must **NEVER** be used for active patient care, diagnosis, or triage during live clinical emergencies.

---

## 📜 License

Copyright (c) 2026 HouseMD Team. All rights reserved.

```

```