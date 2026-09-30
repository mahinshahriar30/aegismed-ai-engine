# src/api.py
import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from src.database import collection, ensure_knowledge_base
from src.engine import build_not_sure_response, diagnose_patient
from src.schema import AegisMedAuditResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aegismed.api")

# --- Configuration -------------------------------------------------------------
# No default key: set AEGIS_API_KEY in Render's environment. If it is missing, every
# request is rejected instead of silently accepting a well-known key.
EXPECTED_API_KEY = os.getenv("AEGIS_API_KEY")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")  # Render sets this automatically

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "https://housemdai.netlify.app").split(",")
    if o.strip()
]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    document_text: str = Field(..., min_length=1, max_length=8000)


# --- Keep-alive (prevents Render free-tier sleep) -----------------------------------
async def keep_alive_loop():
    await asyncio.sleep(15)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/health"
                r = await client.get(url, timeout=10.0)
                logger.info("Keep-alive ping %s -> %s", url, r.status_code)
            except Exception as e:
                logger.warning("Keep-alive ping failed: %s", e)
            await asyncio.sleep(240)


def _load_knowledge_base():
    try:
        logger.info("Knowledge base ready: %d guidelines", ensure_knowledge_base())
    except Exception:
        logger.exception("Could not load the knowledge base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AegisMed AI Engine microservice...")
    if not EXPECTED_API_KEY:
        logger.error("AEGIS_API_KEY is not set: all /api/v1/diagnose requests will be rejected.")
    # Runs in the background so the server opens its port immediately on Render.
    # Until it finishes, requests are answered "Not Sure".
    ingest_task = asyncio.create_task(asyncio.to_thread(_load_knowledge_base))
    task = asyncio.create_task(keep_alive_loop()) if RENDER_EXTERNAL_URL else None
    yield
    if task:
        task.cancel()
    ingest_task.cancel()
    logger.info("Shutting down AegisMed AI Engine microservice...")


app = FastAPI(
    title="AegisMed AI Engine - HouseMD Microservice",
    description="Clinical decision support prototype using guideline-grounded RAG.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not EXPECTED_API_KEY or not api_key or not secrets.compare_digest(api_key, EXPECTED_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key in 'X-API-Key' header.",
        )
    return api_key


# --- Endpoints -----------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    try:
        guideline_count = collection.count()
        rag_status = "initialized" if guideline_count > 0 else "empty"
    except Exception:
        guideline_count, rag_status = 0, "unavailable"
    return {
        "status": "healthy",
        "service": "AegisMed AI Engine",
        "version": "2.1.0",
        "rag_status": rag_status,
        "guideline_chunks": guideline_count,
    }


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so slow LLM calls
# don't freeze the whole server, including /health and the keep-alive loop.
@app.post(
    "/api/v1/diagnose",
    response_model=AegisMedAuditResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_clinical_presentation(
    request: DiagnoseRequest, api_key: str = Security(verify_api_key)
):
    """Always answers 200 with a valid audit: AI result, database fallback, or 'Not Sure'."""
    try:
        return diagnose_patient(document_text=request.document_text)
    except Exception:
        logger.exception("Unexpected error in /api/v1/diagnose")
        return build_not_sure_response("An internal error occurred while analysing this presentation.")