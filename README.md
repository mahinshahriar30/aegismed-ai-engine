<div align="center">

# 🏥 AegisMed AI Engine (HouseMD)
### *Deterministic Dual-Tier Mass Casualty & Emergency Clinical Triage Framework*

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://aegismed-ai-engine-3peyn7pylqupgp2rbdxzwm.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=chromadb&logoColor=white)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)

---

<p align="center">
  <b>A zero-hallucination Clinical Decision Support System (CDSS) designed for Emergency Departments, Rural Diagnostic Centers, and Mass Casualty Incidents.</b>
</p>

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [36 Clinical Guidelines](#-indexed-clinical-guidelines-36-conditions) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation)

</div>

---

## 💡 Overview

High-stress clinical triage requires **absolute diagnostic precision** for critical life-threatening conditions, paired with **flexible reasoning** for non-standard presentations. Standard LLM approaches carry risks of hallucinations or ungrounded recommendations during emergencies.

**AegisMed AI Engine** solves this with a **Dual-Tier RAG Architecture**:
1. **Tier 1 (Strict Grounding):** Forces zero-hallucination compliance against **36 pre-indexed clinical emergency guidelines** stored in ChromaDB.
2. **Tier 2 (General Fallback):** Safely falls back to foundation model reasoning for unindexed presentations, clearly labeling them as general assessments.

---

## ✨ Key Features

* 🎯 **Dual-Tier Zero-Hallucination Engine:** Guarantees strict guideline compliance for critical emergencies (`CRITICAL_EMERGENCY` / `HIGH_PRIORITY`).
* ⚡ **36 High-Yield Regional Guidelines:** Pre-indexed emergency protocols tailored for acute trauma, toxicology, stroke, sepsis, and tropical infectious disasters.
* 🛡️ **Resilient 10-Model Failover Cascade:** Automatically cycles through a fallback sequence of Google Gemini models to handle rate limits and API outages seamlessly.
* 📋 **Strict Type Enforcement:** Native GenAI schema binding with Pydantic (`AegisMedAuditResponse`) ensures 100% reliable JSON output for EHR integration.
* 🌐 **Production Ready:** Microservice backend hosted on **Render** paired with a responsive **Streamlit** frontend dashboard.

---

## 🏗️ System Architecture