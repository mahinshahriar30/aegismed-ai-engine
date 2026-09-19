import json
import os
from google import genai
from google.genai import types

from src.database import query_medical_kb
from src.schema import AegisMedAuditResponse

# Active Gemini 3.x production model endpoints
GEMINI_CASCADE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]


def generate_clinical_audit(prompt: str, client: genai.Client, response_schema):
    """Executes clinical analysis using current Gemini 3.x endpoints.

    Fails over seamlessly without legacy parameters.
    """
    last_exception = None

    for model_name in GEMINI_CASCADE_MODELS:
        try:
            print(f"[Engine] Attempting generation with model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            print(f"[Engine] Successfully generated audit using model: {model_name}")
            return response

        except Exception as e:
            print(f"[Engine] Model '{model_name}' failed: {e}. Falling over...")
            last_exception = e
            continue

    raise RuntimeError(
        f"All models in the failover cascade failed. Last error: {last_exception}"
    )


def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
    """Wrapper function called by api.py to process a patient presentation."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    client = genai.Client(api_key=api_key)

    # 1. Retrieve matched clinical guidelines from ChromaDB
    rag_context = query_medical_kb(document_text)

    # 2. Construct the clinical audit prompt
    prompt = (
        f"VERIFIED CLINICAL GUIDELINES:\n{rag_context}\n\n"
        f"PATIENT PRESENTATION:\n{document_text}\n\n"
        "Provide a complete clinical audit in structured JSON matching the requested schema."
    )

    # 3. Execute LLM call
    response = generate_clinical_audit(
        prompt=prompt,
        client=client,
        response_schema=AegisMedAuditResponse,
    )

    # 4. Parse and return structured response
    return AegisMedAuditResponse.model_validate_json(response.text)