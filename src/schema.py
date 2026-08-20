from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class TriageLevel(str, Enum):
    CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"  # Immediate resuscitation / ICU protocol (Red)
    HIGH_PRIORITY = "HIGH_PRIORITY"            # Urgent ER / specialist intervention needed (Yellow)
    STABLE = "STABLE"                          # Routine clinical management & follow-up (Green)

class DiagnosisItem(BaseModel):
    condition_name: str = Field(
        description="Explicit clinical diagnosis (e.g., 'Acute Ischemic Stroke', 'Acute Myocardial Infarction', 'Diabetic Ketoacidosis')"
    )
    triage_level: TriageLevel = Field(
        description="Emergency triage severity: CRITICAL_EMERGENCY, HIGH_PRIORITY, or STABLE"
    )
    clinical_justification: str = Field(
        description="Medical reasoning linking symptoms, vitals, and lab biomarkers to this diagnosis"
    )
    reference_guideline: str = Field(
        description="The exact clinical criteria or guideline retrieved from the knowledge base"
    )

class AegisMedAuditResponse(BaseModel):
    patient_summary: str = Field(
        description="Concise clinical summary of patient presentation and abnormal findings"
    )
    primary_diagnosis: DiagnosisItem = Field(
        description="The primary, highest-priority clinical diagnosis synthesized from presentation"
    )
    differential_diagnoses: List[DiagnosisItem] = Field(
        default=[],
        description="Secondary or differential diagnoses to consider"
    )
    immediate_emergency_actions: List[str] = Field(
        description="Urgent clinical management steps, diagnostics to order, and stabilization protocols"
    )