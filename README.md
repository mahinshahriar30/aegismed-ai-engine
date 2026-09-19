<div align="center">

# 🏥 AegisMed AI Engine (HouseMD)
### *Deterministic Dual-Tier Mass Casualty & Emergency Clinical Triage Framework*

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-Netlify_CDN-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://housemdai.netlify.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker_Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

---

<p align="center">
  <b>A zero-hallucination Clinical Decision Support System (CDSS) designed for Emergency Departments, Rural Diagnostic Centers, and Mass Casualty Incidents.</b><br>
  👉 <b>Live Application:</b> <a href="https://housemdai.netlify.app">housemdai.netlify.app</a>
</p>

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [36 Clinical Guidelines](#-indexed-clinical-guidelines-36-conditions) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [License](#-license)

</div>

---

## 💡 Overview

High-stress clinical triage requires **absolute diagnostic precision** for critical life-threatening conditions, paired with **flexible reasoning** for non-standard presentations. Standard LLM approaches carry risks of hallucinations or ungrounded recommendations during emergencies.

**AegisMed AI Engine** solves this with a **Dual-Tier RAG Architecture**:
1. **Tier 1 (Strict Grounding):** Forces zero-hallucination compliance against **36 doctor-verified clinical emergency guidelines** stored in ChromaDB.
2. **Tier 2 (General Fallback):** Safely falls back to foundation model reasoning for unindexed presentations, clearly labeling them as general assessments.

---

## ✨ Key Features

* 🎯 **Dual-Tier Zero-Hallucination Engine:** Guarantees strict guideline compliance for critical emergencies (`CRITICAL_EMERGENCY` / `HIGH_PRIORITY`).
* 🩺 **Physician-Reviewed Knowledge Base:** `data/medical_reference.txt` protocols are curated and verified by real medical doctors for clinical accuracy during critical emergencies.
* ⚡ **36 High-Yield Regional Guidelines:** Pre-indexed emergency protocols tailored for acute trauma, toxicology, stroke, sepsis, and tropical infectious disasters.
* 🛡️ **Resilient 10-Model Failover Cascade:** Automatically cycles through a fallback sequence of Google Gemini models to handle rate limits and API outages seamlessly.
* 🔒 **Header-Based Authentication:** Core API endpoints secured with `X-API-Key` validation.
* ⚡ **Sleep-Proof Infrastructure:** Continuous keep-alive heartbeat (`/health` endpoint pinged via external uptime monitor) prevents cold starts on free-tier container deployments.
* 📋 **Strict Type Enforcement:** Native GenAI schema binding with Pydantic (`AegisMedAuditResponse`) ensures 100% reliable JSON output for EHR integration.
* 🌐 **Production Decoupled Setup:** High-performance static web app hosted on **Netlify CDN** paired with a containerized **Render Docker** backend microservice.

---

## 🏗️ System Architecture

```text
 ┌────────────────────────┐              ┌──────────────────────────────────┐
 │  Netlify CDN Frontend  │              │  Incoming Clinical Presentation   │
 │   (src/index.html)     │              └────────────────┬─────────────────┘
 └───────────┬────────────┘                               │
             │ CORS / X-API-Key                           │
             ▼                                            ▼
 ┌────────────────────────┐              ┌──────────────────────────────────┐
 │ Render Docker Backend  │─────────────>│       ChromaDB Vector Query      │
 │  (FastAPI Microservice)│              └────────────────┬─────────────────┘
 └────────────────────────┘                               │
                                  ┌───────────────────────┴───────────────────────┐
                                  ▼                                               ▼
                     [ Match Found in Vector Index ]                 [ No Match Found / Stable ]
                     (36 Doctor-Reviewed Guidelines)                 (Unindexed Presentation)
                                  │                                               │
                                  ▼                                               ▼
                   ┌──────────────────────────────┐                ┌──────────────────────────────┐
                   │   TIER 1: STRICT GROUNDING   │                │   TIER 2: GENERAL FALLBACK   │
                   │  • Zero-Hallucination Rule   │                │  • Foundation Model Reasoning│
                   │  • Grounded Step-by-Step Plan│                │  • Flagged as General Case   │
                   │  • High-Priority Action      │                │  • Lower Triage Priority     │
                   └──────────────────────────────┘                └──────────────────────────────┘

```

---

## 📚 Indexed Clinical Guidelines (36 Conditions)

> 🩺 **Medical Review Note:** All 36 clinical reference protocols in `data/medical_reference.txt` have been rigorously curated and reviewed by practicing emergency physicians to ensure adherence to standard acute care, resuscitation, and toxicological guidelines.

| Category | Indexed Emergency Conditions |
| --- | --- |
| **Cardiovascular & Cerebrovascular** | Acute Ischemic/Hemorrhagic Stroke, Acute Coronary Syndrome (STEMI/NSTEMI), Acute Pulmonary Embolism, Hypertensive Emergency, Acute Decompensated Heart Failure |
| **Trauma & Resuscitation** | Traumatic Hemorrhagic Shock, Polytrauma / Road Traffic Accident, Traumatic Brain Injury (TBI/EDH), Tension Pneumothorax, Cardiac Arrest (ACLS), Drowning Asphyxia |
| **Toxico-Environmental** | Organophosphate Toxicity, Paracetamol Overdose, Benzodiazepine Overdose, Aluminum Phosphide (Rice Tablet), Methanol Poisoning, Carbon Monoxide Poisoning, Heat Stroke, Snakebite Envenomation |
| **Sepsis & Infectious Disasters** | Severe Septic Shock, Dengue Hemorrhagic Fever (DHF), Severe Falciparum Malaria, Acute Watery Diarrhea / Cholera Shock, Acute Bacterial Meningitis, Rabies Exposure |
| **Obstetrics & Metabolic** | Eclampsia / Severe Preeclampsia, Diabetic Ketoacidosis (DKA/HHS), Status Epilepticus, Acute Kidney Injury (AKI), Acute Pancreatitis, Upper GI Bleeding, Perforated Peptic Ulcer, Acute Liver Failure, Severe Acute Malnutrition (SAM) |

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone [https://github.com/mahinshahriar30/aegismed-ai-engine.git](https://github.com/mahinshahriar30/aegismed-ai-engine.git)
cd aegismed-ai-engine

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
AEGIS_API_KEY="aegismed_secure_key_2026"
PORT=10000

```

### 3. Run via Docker Locally

```bash
# Build Docker container
docker build -t aegismed-ai-engine .

# Run container
docker run -p 10000:10000 --env-file .env aegismed-ai-engine

```

> **Interactive API Docs:** Access Swagger UI at `http://localhost:10000/docs`

---

## 📡 API Documentation

### 1. Health Check & Heartbeat (Unauthenticated)

```http
GET /health

```

**Response:**

```json
{
  "status": "healthy",
  "service": "AegisMed AI Engine",
  "version": "2.0.0",
  "rag_status": "initialized"
}

```

### 2. Clinical Diagnosis Audit

```http
POST /api/v1/diagnose
Headers:
  X-API-Key: aegismed_secure_key_2026
  Content-Type: application/json

```

**Sample Request Body:**

```json
{
  "document_text": "EMERGENCY PRESENTATION: 28-year-old agricultural worker presenting with sudden pinpoint pupils (miosis), profuse salivation, vomiting, wheezing, and bradycardia (HR 48 bpm, BP 85/55 mmHg) after crop spraying."
}

```

**Sample Response:**

```json
{
  "patient_summary": "28-year-old male presenting with cholinergic crisis symptoms following agricultural pesticide exposure.",
  "primary_diagnosis": {
    "condition_name": "Acute Organophosphate Insecticide Toxicity",
    "triage_level": "CRITICAL_EMERGENCY",
    "clinical_justification": "Presentation matches classic cholinergic crisis criteria (DUMBBELSS: miosis, salivation, bronchospasm, bradycardia, hypotension).",
    "reference_guideline": "ORGANOPHOSPHATE_POISONING_TOXICOLOGY"
  },
  "differential_diagnoses": [],
  "immediate_emergency_actions": [
    "Remove contaminated clothing and wash skin with soap and cold water immediately (staff PPE required)",
    "Administer Atropine 2 mg to 5 mg IV bolus every 5-10 minutes until full atropinization (clearing of lungs, HR > 80 bpm)",
    "Administer Pralidoxime (2-PAM) 1 g to 2 g IV over 15-30 minutes followed by continuous infusion",
    "Prepare for early endotracheal intubation if respiratory failure or bronchorrhea persists"
  ]
}

```

---

## 📂 Project Structure

```text
aegismed-ai-engine/
├── data/
│   └── medical_reference.txt  # 36 doctor-reviewed clinical emergency guidelines
├── src/
│   ├── api.py                 # FastAPI microservice, CORS policy & keep-alive loop
│   ├── database.py            # ChromaDB persistent vector database manager
│   ├── engine.py              # Dual-tier prompt engine & 10-model failover cascade
│   ├── index.html             # Netlify single-page application frontend
│   └── schema.py              # Pydantic contracts & triage data enums
├── .env                       # Local environment variables configuration
├── .gitignore                 # Exclusion rules for secrets & build outputs
├── Dockerfile                 # Multi-stage security-hardened non-root Docker build
├── netlify.toml               # Netlify CDN deployment & security headers config
├── requirements.txt           # Core Python dependencies
├── test_db.py                 # Vector DB retrieval integration test
└── test_engine.py             # Diagnostic pipeline unit test suite

```

---

## ⚠️ Research Disclaimer

> **RESEARCH PROTOTYPE ONLY:** AegisMed AI Engine (HouseMD) is an experimental software proof-of-concept created strictly for technical demonstration, software benchmarking, and academic research evaluation. It is **NOT** a certified medical device and must **NEVER** be used for active patient care, diagnosis, or triage during real clinical emergencies.

---

## 📜 License

Copyright (c) 2026 Mahin Shahriar. All rights reserved.

```
