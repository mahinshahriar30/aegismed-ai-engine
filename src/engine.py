import logging
import os
from typing import List
from google import genai
from google.genai import types
from src.database import query_medical_kb
from src.schema import AegisMedAuditResponse

# Suppress harmless SDK logs
logging.getLogger("google_genai").setLevel(logging.ERROR)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GEMINI_API_KEY environment variable is missing!")

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
You are AegisMed AI Engine, an expert emergency clinical decision support system optimized for acute care and mass casualty triage.

Your objective is to analyze medical admission notes, clinical reports, lab results, and patient symptoms, then synthesize them into a structured medical diagnostic report following standard emergency triage protocols.

OPERATIONAL DUAL-TIER SAFETY PROTOCOL:

1. TIER 1 — STRICT VERIFIED GUIDELINE MATCHING (Critical / High-Priority Emergencies):
   - First, evaluate whether the patient's presentation matches any condition in the provided Clinical Reference Guidelines.
   - If a match is found in the guidelines, you MUST ground your primary diagnosis, immediate emergency actions, and clinical justification STRICTLY on the retrieved context.
   - DO NOT extrapolate, infer unverified treatments, or hallucinate beyond the provided guidelines for Tier 1 matches.

2. TIER 2 — GENERAL EMERGENCY REASONING FALLBACK (Unindexed / Low-Priority Cases):
   - If the presentation DOES NOT match any condition in the provided Clinical Reference Guidelines, leverage your general medical foundation knowledge to evaluate the patient.
   - Explicitly indicate in the reference_guideline field that no direct match was found in the indexed emergency protocols, and tag the response as a General Clinical Assessment.
   - Assign appropriate triage levels based on patient severity to prevent misclassification.

TRIAGE LEVEL CLASSIFICATION:
- CRITICAL_EMERGENCY: Immediate life-threatening conditions (e.g., Acute Stroke, MI, Respiratory Failure, Septic Shock, Severe Toxicity/Poisoning).
- HIGH_PRIORITY: Urgent conditions requiring rapid clinical/laboratory evaluation and monitoring (e.g., Acute Pancreatitis, AKI, High Fever with Unknown Etiology).
- STABLE: Non-emergent, routine clinical management or non-critical presentations outside disaster scope.
"""

# Valid production Gemini models in order of speed & reasoning quality
PREFERRED_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
  # 1. Retrieve guidelines from vector DB
  retrieved_context = query_medical_kb(document_text, n_results=3)

  context_prompt = f"""
    [RETRIEVED CLINICAL GUIDELINES & REFERENCE DATA]
    {retrieved_context}

    [PATIENT PRESENTATION / MEDICAL REPORT]
    {document_text}
    """

  last_error = None

  # 2. Resilient Failover Loop across valid models
  for model_name in PREFERRED_MODELS:
    try:
      response = client.models.generate_content(
          model=model_name,
          contents=context_prompt,
          config=types.GenerateContentConfig(
              system_instruction=SYSTEM_INSTRUCTION,
              response_mime_type="application/json",
              response_schema=AegisMedAuditResponse,
              temperature=0.1,
          ),
      )

      # Validate and return Pydantic schema
      return AegisMedAuditResponse.model_validate_json(response.text)

    except Exception as err:
      last_error = err
      print(
          f"⚠️ Model '{model_name}' failed ({type(err).__name__}: {err}). Trying"
          " next fallback model..."
      )
      continue

  # 3. If all model fallbacks fail
  print(f"❌ All model fallbacks failed. Last error: {last_error}")
  raise last_error