from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.schema import AegisMedAuditResponse
from src.engine import diagnose_patient
from src.database import initialize_medical_kb

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting AegisMed AI Engine...")
    initialize_medical_kb()
    yield

app = FastAPI(
    title="AegisMed AI - Emergency Medical Diagnostic Engine",
    version="2.0.0",
    description="Enterprise AI Medical Engine specialized in high-precision clinical diagnosis and emergency triage.",
    lifespan=lifespan
)

class PatientRequest(BaseModel):
    document_text: str = Field(..., description="Patient clinical report, lab results, or presentation notes.")
    domain: Optional[str] = Field(default="medical", description="Optional domain field for backward compatibility.")

# Supports both /api/v1/diagnose and /api/v1/audit endpoints
@app.post("/api/v1/diagnose", response_model=AegisMedAuditResponse)
@app.post("/api/v1/audit", response_model=AegisMedAuditResponse)
def evaluate_patient(payload: PatientRequest):
    if not payload.document_text.strip():
        raise HTTPException(status_code=400, detail="Patient document text cannot be empty.")
    try:
        return diagnose_patient(payload.document_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "AegisMed Engine Operational"}