import asyncio
from contextlib import asynccontextmanager
import os
import httpx
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from src.database import RAGDatabase
from src.engine import AegisMedEngine
from src.schema import AegisMedAuditRequest, AegisMedAuditResponse

# 1. Configuration & Security Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

EXPECTED_API_KEY = os.getenv("AEGIS_API_KEY", "aegismed_secure_key_2026")
RENDER_EXTERNAL_URL = os.getenv(
    "RENDER_EXTERNAL_URL", ""
)  # Render automatically sets this env var


async def keep_alive_loop():
  """Background worker to ping /health every 10 minutes to keep Render instance warm."""
  # Wait 30 seconds after startup before starting the loop
  await asyncio.sleep(30)

  async with httpx.AsyncClient() as client:
    while True:
      try:
        # If RENDER_EXTERNAL_URL is available, ping public URL; otherwise fallback to local port
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

      # Wait 10 minutes (600 seconds) between pings
      await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
  """Application lifecycle manager to initialize RAG database and keep-alive task."""
  print("Starting AegisMed AI Engine microservice...")

  # Initialize ChromaDB vector database
  db = RAGDatabase()
  db.initialize_reference_data()

  # Instantiate core reasoning engine
  app.state.engine = AegisMedEngine(db=db)

  # Start keep-alive ping loop in the background
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
    allow_origins=["*"],  # Permits Streamlit Cloud and local frontends
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
  """Health check endpoint used by Render load balancer and self-ping keep-alive loop."""
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
async def analyze_clinical_presentation(
    request: AegisMedAuditRequest,
    # api_key: str = Depends(verify_api_key) # Uncomment to enforce API Key auth
):
  """Primary endpoint to execute clinical triage against ChromaDB RAG guidelines."""
  try:
    engine: AegisMedEngine = app.state.engine
    audit_result = engine.audit_document(
        document_text=request.document_text, domain=request.domain
    )
    return audit_result
  except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Clinical analysis engine error: {str(e)}",
    )