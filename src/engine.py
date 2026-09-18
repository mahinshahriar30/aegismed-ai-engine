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

# Backup list of verified Gemini 3.x series models
STATIC_FALLBACK_POOL: List[str] = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-preview",
    "gemini-3.0-flash",
    "gemini-3.0-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash"
]

def get_exact_10_valid_models() -> List[str]:
    """Dynamically queries the API for active text models and selects exactly 10 valid candidates."""
    valid_models = []
    
    try:
        # Fetch models actively supported by your key
        for m in client.models.list():
            model_id = m.name.replace("models/", "")
            methods = getattr(m, "supported_generation_methods", [])

            # Filter for text models only (exclude video, audio, live streaming, robotics)
            if "generateContent" in methods and not any(
                tag in model_id for tag in ["live", "audio", "veo", "translate", "robotics", "realtime", "transcribe"]
            ):
                valid_models.append(model_id)
    except Exception as err:
        print(f"⚠️ Dynamic model discovery failed: {err}")

    # Fill remaining slots using static fallbacks if API list returns fewer than 10
    for fallback in STATIC_FALLBACK_POOL:
        if fallback not in valid_models:
            valid_models.append(fallback)

    # Slice to strictly 10 models
    final_10 = valid_models[:10]
    print(f"[Model Discovery] Selected 10 active failover models: {final_10}")
    return final_10

ACTIVE_MODELS = get_exact_10_valid_models()

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

    # 2. Iterate through all 10 failover models until one succeeds
    for idx, model_name in enumerate(ACTIVE_MODELS, start=1):
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
            print(f"⚠️ Model {idx}/10 ('{model_name}') failed ({type(err).__name__}: {err}). Trying next model...")
            continue

    # 3. Raise exception if all 10 models fail
    print(f"❌ All 10 model fallbacks failed. Last error: {last_error}")
    raise last_error