import os
import logging
from typing import List
from google import genai
from google.genai import types
from src.schema import AegisMedAuditResponse
from src.database import query_medical_kb

# Suppress harmless SDK info logs
logging.getLogger("google_genai").setLevel(logging.ERROR)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is missing!")

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
You are AegisMed AI, an expert emergency clinical decision support engine.
Your objective is to analyze medical admission notes, clinical reports, lab results, and patient symptoms,
then synthesize them into a structured medical diagnostic report following standard triage protocols.

Always rely on the retrieved clinical reference guidelines to justify the primary diagnosis and triage level.
Assign appropriate triage levels:
- CRITICAL_EMERGENCY: Immediate life-threatening conditions (e.g., Acute Stroke, MI, Pulmonary Embolism, Septic Shock, Poisoning, Snakebite).
- HIGH_PRIORITY: Urgent conditions needing rapid clinical/laboratory evaluation (e.g., Acute Pancreatitis, AKI).
- STABLE: Non-emergent, routine clinical management.
"""

# Ordered list of 10 fast and reliable models from your available list
PREFERRED_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
    "gemini-3.7-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest"
]

def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
    # 1. Retrieve guidelines from ChromaDB
    retrieved_context = query_medical_kb(document_text, n_results=3)
    
    context_prompt = f"""
    [RETRIEVED CLINICAL GUIDELINES & REFERENCE DATA]
    {retrieved_context}

    [PATIENT PRESENTATION / MEDICAL REPORT]
    {document_text}
    """

    last_error = None

    # 2. Resilient Failover Loop across all 10 models
    for model_name in PREFERRED_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=context_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=AegisMedAuditResponse,
                    temperature=0.1
                )
            )
            
            # Parse and return structured response
            return AegisMedAuditResponse.model_validate_json(response.text)

        except Exception as err:
            last_error = err
            print(f"⚠️ Model '{model_name}' failed ({type(err).__name__}). Trying next fallback model...")
            continue

    # 3. If all 10 model fallbacks fail
    print("❌ All 10 model fallbacks failed.")
    raise last_error