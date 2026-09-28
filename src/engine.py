# src/engine.py

import json
import os
from google import genai
from google.genai import types
from groq import Groq

from src.database import query_medical_kb, collection
from src.schema import AegisMedAuditResponse

# Cascade list using active API identifiers
CASCADE_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "gemini-2.5-flash"
]


def search_kb_fallback(query_text: str):
    """
    Direct ChromaDB lookup used as a safety net when all LLMs fail.
    Extracts condition and triage data directly from local vector metadata.
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=1,
            include=["metadatas"]
        )
        if results and results.get("metadatas") and len(results["metadatas"]) > 0 and len(results["metadatas"][0]) > 0:
            metadata = results["metadatas"][0][0]
            
            actions_raw = metadata.get("recommended_actions", "")
            if isinstance(actions_raw, str):
                actions = [a.strip() for a in actions_raw.split(";") if a.strip()]
            else:
                actions = actions_raw

            return {
                "condition_name": metadata.get("condition_name", "Unverified Match"),
                "triage_level": metadata.get("triage_level", "Requires Clinical Assessment"),
                "actions": actions or ["Consult standard healthcare protocols."]
            }
    except Exception as e:
        print(f"[Engine] Fallback ChromaDB search error: {e}")
    
    return None


def generate_clinical_audit(prompt: str, response_schema) -> str | None:
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
    groq_client = Groq(api_key=groq_key) if groq_key else None

    for model_name in CASCADE_MODELS:
        # --- Provider 1: Groq ---
        if model_name.startswith("groq/"):
            if not groq_client:
                print(f"[Engine] Skipping {model_name}: GROQ_API_KEY not set.")
                continue
            
            real_groq_model = model_name.replace("groq/", "")
            try:
                schema_json = json.dumps(response_schema.model_json_schema())
                groq_prompt = (
                    f"{prompt}\n\n"
                    f"CRITICAL REQUIREMENT: Return ONLY valid raw JSON matching this schema:\n{schema_json}"
                )
                
                completion = groq_client.chat.completions.create(
                    model=real_groq_model,
                    messages=[
                        {"role": "system", "content": "You are AegisMed AI. Output strictly raw JSON matching the provided schema."},
                        {"role": "user", "content": groq_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                print(f"[Engine] Successfully generated audit using Groq model: {real_groq_model}")
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[Engine] Groq model '{real_groq_model}' failed: {e}")
                continue

        # --- Provider 2: Gemini ---
        else:
            if not gemini_client:
                print(f"[Engine] Skipping {model_name}: GEMINI_API_KEY not set.")
                continue
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    ),
                )
                print(f"[Engine] Successfully generated audit using Gemini model: {model_name}")
                return response.text
            except Exception as e:
                print(f"[Engine] Gemini model '{model_name}' failed: {e}")
                continue

    return None


def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
    try:
        # STEP 1: Mandatory ChromaDB Grounding Search (Anti-Hallucination)
        rag_context = query_medical_kb(document_text)
        has_grounded_context = bool(rag_context and rag_context != "No matching clinical guidelines found.")

        prompt = (
            f"VERIFIED CLINICAL GUIDELINES:\n{rag_context}\n\n"
            f"PATIENT PRESENTATION:\n{document_text}\n\n"
            "Provide a complete clinical audit in structured JSON matching the requested schema."
        )

        # STEP 2: Execute LLM Cascade
        raw_json_string = generate_clinical_audit(
            prompt=prompt,
            response_schema=AegisMedAuditResponse,
        )

        # STEP 3: Emergency Fallback if all LLMs are unavailable/outdated
        if not raw_json_string:
            print("[Engine] All LLMs failed. Performing local ChromaDB fallback search...")
            fallback = search_kb_fallback(document_text)

            if fallback and has_grounded_context:
                return AegisMedAuditResponse(
                    triage_level=fallback["triage_level"],
                    primary_condition=fallback["condition_name"],
                    clinical_summary=(
                        "AI models are temporarily offline. "
                        f"Retrieved primary condition ('{fallback['condition_name']}') and triage status directly from verified local guidelines."
                    ),
                    recommended_actions=fallback["actions"],
                    confidence_score=0.5
                )
            else:
                return AegisMedAuditResponse(
                    triage_level="Not Sure",
                    primary_condition="Not Sure / Unverified Condition",
                    clinical_summary="AI providers are unavailable, and no matching guideline was found in local storage.",
                    recommended_actions=[
                        "Immediate manual clinical evaluation required.",
                        "Consult local medical guidelines.",
                        "Retry request when AI services recover."
                    ],
                    confidence_score=0.0
                )

        # STEP 4: Parse & Return LLM Output
        cleaned_string = raw_json_string.strip()
        if cleaned_string.startswith("```"):
            lines = cleaned_string.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_string = "\n".join(lines).strip()

        return AegisMedAuditResponse.model_validate_json(cleaned_string)

    except Exception as e:
        print(f"[Engine] Emergency fallback triggered due to exception: {e}")
        return AegisMedAuditResponse(
            triage_level="Not Sure",
            primary_condition="Not Sure / System Fallback",
            clinical_summary="Unable to generate AI synthesis. Please review using clinical guidelines.",
            recommended_actions=[
                "Manual clinical review required.",
                "Verify system logs."
            ],
            confidence_score=0.0
        )