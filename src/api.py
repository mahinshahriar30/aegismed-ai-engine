import asyncio
from contextlib import asynccontextmanager
import os
import httpx
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Imports matching your exact codebase
from src.engine import diagnose_patient
from src.schema import AegisMedAuditResponse

# 1. Configuration & Security Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

EXPECTED_API_KEY = os.getenv("AEGIS_API_KEY", "aegismed_secure_key_2026")
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")


class DiagnoseRequest(BaseModel):
  document_text: str


async def keep_alive_loop():
  """Background task to ping /health every 10 minutes to prevent Render free-tier sleeping."""
  await asyncio.sleep(30)  # Initial wait before starting heartbeat

  async with httpx.AsyncClient() as client:
    while True:
      try:
        target_url = (
            f"{RENDER_EXTERNAL_URL}/health"
            if RENDER_EXTERNAL_URL
            else "http://127.0.0.1:8000/health"
        )
        response = await client.get(target_url, timeout=10.0)
        print(
            f"[Keep-Alive] Heartbeat ping sent to {target_url} - Status:"
            f" {response.status_code}"
        )
      except Exception as e:
        print(f"[Keep-Alive] Heartbeat ping failed: {e}")

      await asyncio.sleep(600)  # Ping every 10 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Application lifecycle manager to initialize background keep-alive loop."""
  print("Starting AegisMed AI Engine microservice...")

  # Start background keep-alive task
  keep_alive_task = asyncio.create_task(keep_alive_loop())

  yield

  # Cleanup on shutdown
  keep_alive_task.cancel()
  print("Shutting down AegisMed AI Engine microservice...")


# 2. Instantiate FastAPI Application
app = FastAPI(
    title="AegisMed AI Engine - HouseMD Microservice",
    description=(
        "Production-grade Clinical Decision Support System backend utilizing"
        " dual-tier zero-hallucination RAG."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# 3. Configure CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. Authentication Middleware Dependency
async def verify_api_key(api_key: str = Security(api_key_header)):
  if api_key != EXPECTED_API_KEY:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key in 'X-API-Key' header.",
    )
  return api_key


# 5. Core API Endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
  """Health check endpoint used by Render load balancer and keep-alive loop."""
  return {
      "status": "healthy",
      "service": "AegisMed AI Engine",
      "version": "2.0.0",
      "rag_status": "initialized",
  }


@app.post(
    "/api/v1/diagnose",
    response_model=AegisMedAuditResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_clinical_presentation(request: DiagnoseRequest):
  """Primary endpoint to execute clinical triage against ChromaDB RAG guidelines."""
  try:
    audit_result = diagnose_patient(document_text=request.document_text)
    return audit_result
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Clinical analysis engine error: {str(e)}",
    )