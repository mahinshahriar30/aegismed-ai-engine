# src/engine.py

import json
import os
from google import genai
from google.genai import types
from groq import Groq

from src.database import search_medical_kb_direct, query_medical_kb
from src.schema import AegisMedAuditResponse

# Priority cascade list of operational models
CASCADE_MODELS = [
    #"groq/openai/gpt-oss-20b",
    "gemini-3.5-flash",
    "groq/llama-3.3-70b-versatile"
]


def generate_clinical_audit(prompt: str, response_schema) -> str | None:
    """
    Tries models in CASCADE_MODELS sequentially.
    Returns raw JSON string on success, or None if all fail.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    gemini_client = genai.Client(api_key=gemini_key) if gemini_key else None
    groq_client = Groq(api_key=groq_key) if groq_key else None

    for model_name in CASCADE_MODELS:
        # --- Provider 1: Groq ---
        if model_name.startswith("groq/"):
            if not groq_client:
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
                print(f"[Engine] Successfully generated audit using Groq: {real_groq_model}")
                return completion.choices[0].message.content
            except Exception as e:
                print(f"[Engine] Groq model '{real_groq_model}' failed: {e}")
                continue

        # --- Provider 2: Gemini ---
        else:
            if not gemini_client:
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
                print(f"[Engine] Successfully generated audit using Gemini: {model_name}")
                return response.text
            except Exception as e:
                print(f"[Engine] Gemini model '{model_name}' failed: {e}")
                continue

    # Return None if every single LLM call failed
    return None


def diagnose_patient(document_text: str) -> AegisMedAuditResponse:
    """
    Execution Strategy:
    1. Direct ChromaDB Check: If vector distance < threshold, return direct match immediately.
    2. RAG Context Retrieval: Retrieve relevant medical guideline excerpts from ChromaDB.
    3. Multi-LLM Cascade: Query LLM models in order.
    4. Fallback Handling: Return structured error response if all providers are unavailable.
    """
    
    # --- STEP 1: Direct Matching in ChromaDB ---
    direct_match = search_medical_kb_direct(document_text, similarity_threshold=0.90)
    
    if direct_match:
        print("[Engine] Direct ChromaDB match found! Bypassing LLM generation.")
        return AegisMedAuditResponse(
            triage_level=direct_match["triage_level"],
            primary_condition=direct_match["condition_name"],
            clinical_summary=f"Direct database match identified: {direct_match['condition_name']}.",
            recommended_actions=direct_match.get(
                "actions", 
                [f"Follow standard clinical protocol for {direct_match['condition_name']}."]
            ),
            confidence_score=1.0
        )

    # --- STEP 2: RAG Context Retrieval & LLM Generation ---
    rag_context = query_medical_kb(document_text)
    
    prompt = (
        f"VERIFIED CLINICAL GUIDELINES:\n{rag_context}\n\n"
        f"PATIENT PRESENTATION:\n{document_text}\n\n"
        "Provide a complete clinical audit in structured JSON matching the requested schema."
    )

    raw_json_string = generate_clinical_audit(
        prompt=prompt,
        response_schema=AegisMedAuditResponse,
    )

    # --- STEP 3: Fallback Message if All Models Fail ---
    if not raw_json_string:
        print("[Engine] All LLM models failed. Returning fallback response.")
        return AegisMedAuditResponse(
            triage_level="Requires Urgent Human Review",
            primary_condition="Service Unavailable / Clinical Audit Incomplete",
            clinical_summary="All AI generation models are currently experiencing high demand or maintenance. "
                             "Please review verified guidelines directly or retry the request shortly.",
            recommended_actions=[
                "Perform standard manual triage protocol.",
                "Consult local clinical guideline repository.",
                "Retry API request in 1-2 minutes."
            ],
            confidence_score=0.0
        )

    # --- STEP 4: Clean Markdown Blocks & Return Pydantic Model ---
    cleaned_string = raw_json_string.strip()
    if cleaned_string.startswith("```"):
        lines = cleaned_string.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_string = "\n".join(lines).strip()

    return AegisMedAuditResponse.model_validate_json(cleaned_string)