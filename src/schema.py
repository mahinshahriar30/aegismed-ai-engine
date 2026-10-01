# src/schema.py
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TriageLevel(str, Enum):
    CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"  # Immediate resuscitation / ICU protocol (Red)
    HIGH_PRIORITY = "HIGH_PRIORITY"            # Urgent ER / specialist intervention (Yellow)
    STABLE = "STABLE"                          # Routine management & follow-up (Green)
    UNDETERMINED = "UNDETERMINED"              # Not sure: manual triage required (Grey)


_NON = {"NON"}
_CRITICAL = {"CRITICAL", "EMERGENCY", "RED", "IMMEDIATE", "RESUSCITATION"}
_STABLE = {"STABLE", "ROUTINE", "GREEN", "LOW"}
_HIGH = {"HIGH", "URGENT", "YELLOW", "EMERGENT"}


def normalize_triage(value) -> TriageLevel:
    """
    Maps any triage string (from the LLM or from ChromaDB metadata) onto the enum.
    Anything we cannot confidently map becomes UNDETERMINED, never a guess.
    """
    if isinstance(value, TriageLevel):
        return value
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if text in TriageLevel.__members__:
        return TriageLevel[text]

    tokens = set(text.split("_"))
    if tokens & _NON:  # "non-urgent", "non-critical"
        return TriageLevel.STABLE
    if tokens & _CRITICAL:
        return TriageLevel.CRITICAL_EMERGENCY
    if tokens & _STABLE:
        return TriageLevel.STABLE
    if tokens & _HIGH:
        return TriageLevel.HIGH_PRIORITY
    return TriageLevel.UNDETERMINED


class DiagnosisItem(BaseModel):
    condition_name: str = Field(
        description="Explicit clinical diagnosis taken from the provided guidelines, or 'Not Sure'"
    )
    triage_level: TriageLevel = Field(
        description="Exactly one of: CRITICAL_EMERGENCY, HIGH_PRIORITY, STABLE, UNDETERMINED"
    )
    clinical_justification: str = Field(
        description="Reasoning linking symptoms, vitals and labs to this diagnosis, using only the provided guidelines"
    )
    reference_guideline: str = Field(
        description="Guideline title(s) the diagnosis is based on"
    )

    @field_validator("triage_level", mode="before")
    @classmethod
    def _coerce_triage(cls, v):
        return normalize_triage(v)


class LLMAuditOutput(BaseModel):
    """What the LLM is asked to produce. Fields controlled by our code are NOT in here."""

    patient_summary: str = Field(
        description="Concise clinical summary of the presentation and abnormal findings"
    )
    primary_diagnosis: DiagnosisItem = Field(
        description="The primary, highest-priority diagnosis supported by the guidelines"
    )
    differential_diagnoses: List[DiagnosisItem] = Field(
        default_factory=list,
        description="Secondary diagnoses supported by the guidelines",
    )
    immediate_emergency_actions: List[str] = Field(
        description="Urgent management steps taken from the guidelines"
    )


class AegisMedAuditResponse(LLMAuditOutput):
    """Final API response. The extra fields are always set by our code, never by the LLM."""

    source: Literal["ai_grounded", "database_fallback", "no_match"] = "ai_grounded"
    grounded: bool = True
    reference_guidelines: List[str] = Field(default_factory=list)
    notice: Optional[str] = None