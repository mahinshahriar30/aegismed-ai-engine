import json
import os
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError
from groq import Groq

from src.database import query_medical_kb
from src.schema import AegisMedAuditResponse

# Active endpoints verified on Google & Groq
CASCADE_MODELS = [
    # 1. Primary Gemini models (Stable & Active)
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",

    # 2. Active Groq Production Backups
    "groq/llama-3.1-8b-instant",
    "groq/openai/gpt-oss-20b"
]


def generate_clinical_audit(prompt: str, response_schema) -> str:
    """Executes clinical analysis using Gemini endpoints and fails over to Groq."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
    groq_client = Groq(api_key=groq_key) if groq_key else None

    last_exception = None

    for model_name in CASCADE_MODELS:
        # --- Handle Groq Fallback ---
        if model_name.startswith("groq/"):
            if not groq_client:
                print("[Engine] GROQ_API_KEY missing or not initialized. Skipping Groq.")
                continue

            real_groq_model = model_name.replace("groq/", "")
            try:
                print(f"[Engine] Attempting generation with Groq model: {real_groq_model}")

                schema_json = json.dumps(response_schema.model_json_schema())
                groq_prompt = (
                    f"{prompt}\n\n"
                    f"CRITICAL REQUIREMENT: Return ONLY valid raw JSON matching this schema:\n{schema_json}"
                )

                completion = groq_client.chat.completions.create(
                    model=real_groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are AegisMed AI, an expert clinical triage engine. Output strictly raw JSON matching the provided schema."
                        },
                        {
                            "role": "user",
                            "content": groq_prompt
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=2048
                )

                raw_text = completion.choices[0].message.content
                print(f"[Engine] Successfully generated audit using Groq: {real_groq_model}")
                return raw_text

            except Exception as e:
                print(f"[Engine] Groq model '{real_groq_model}' failed: {e}. Falling over...")
                last_exception = e
                continue

        # --- Handle Gemini Models ---
        else:
            if not gemini_client:
                print("[Engine] GEMINI_API_KEY missing. Skipping Gemini.")
                continue

            # Retries for temporary 503 capacity issues
            for attempt in range(2):
                try:
                    print(f"[Engine] Attempting generation with Gemini model: {model_name} (Attempt {attempt + 1})")

                    response = gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_schema,
                        ),
                    )
                    print(f"[Engine] Successfully generated audit using Gemini: {model_name}")
                    return response.text

                except APIError as e:
                    err_str = str(e)
                    print(f"[Engine] Gemini model '{model_name}' failed: {err_str}.")
                    last_exception = e

                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(2)  # Short pause before retry or failover
                        continue
                    break
                except Exception as e:
                    print(f"[Engine] Unexpected failure on '{model_name}': {e}")
                    last_exception = e
                    break

    raise RuntimeError(
        f"All models in the failover cascade failed. Last error: {last_exception}"
    )


def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
    """Wrapper function called by api.py to process a patient presentation."""
    # 1. Retrieve clinical guidelines from ChromaDB
    rag_context = query_medical_kb(document_text)

    # 2. Construct prompt
    prompt = (
        f"VERIFIED CLINICAL GUIDELINES:\n{rag_context}\n\n"
        f"PATIENT PRESENTATION:\n{document_text}\n\n"
        "Provide a complete clinical audit in structured JSON matching the requested schema."
    )

    # 3. Call failover cascade
    raw_json_string = generate_clinical_audit(
        prompt=prompt,
        response_schema=AegisMedAuditResponse,
    )

    # 4. Clean Markdown formatting
    cleaned_string = raw_json_string.strip()
    if cleaned_string.startswith("```"):
        lines = cleaned_string.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_string = "\n".join(lines).strip()

    # 5. Parse into Pydantic model
    return AegisMedAuditResponse.model_validate_json(cleaned_string)