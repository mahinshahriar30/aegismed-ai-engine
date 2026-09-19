import os
from google import genai
from google.genai import types

# 10 High-Availability Gemini Models (Ordered by Speed & Reliability)
GEMINI_CASCADE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.5-flash-lite",
]


def generate_clinical_audit(prompt: str, client: genai.Client, response_schema):
    """Executes clinical analysis directly using the fastest model in the cascade.

    Automatically fails over to backup models on rate-limits, timeouts, or quota errors.
    Zero extra discovery latency.
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
                    temperature=0.1,  # Low variance for clinical determinism
                ),
            )
            print(f"[Engine] Successfully generated audit using model: {model_name}")
            return response

        except Exception as e:
            print(f"[Engine] Model '{model_name}' failed or rate-limited: {e}. Falling over...")
            last_exception = e
            continue

    raise RuntimeError(
        f"All 10 models in the failover cascade failed. Last error: {last_exception}"
    )