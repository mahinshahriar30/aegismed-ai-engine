"""
End-to-end test of the diagnose pipeline. Run from the project root:

    python test_engine.py

Without GROQ_API_KEY set, matched cases are answered from the database (database_fallback).
With it set, they are answered by the AI (ai_grounded). The unrelated case must always be no_match.
"""
import sys

from src.database import ensure_knowledge_base
from src.engine import diagnose_patient

# (name, presentation, expected top guideline or None when "Not Sure" is expected)
CASES = [
    (
        "Stroke",
        "PATIENT EMERGENCY ADMISSION: 62-year-old male presenting with sudden right-sided facial drooping, "
        "slurred speech, and weakness in the right arm. BP: 195/110 mmHg. Onset: 90 minutes ago.",
        "ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE",
    ),
    (
        "Benzodiazepine overdose",
        "EMERGENCY ADMISSION: 29-year-old female brought in unresponsive next to empty pill bottles of "
        "Alprazolam (Xanax). SpO2 86% on room air, RR 8/min (severe respiratory depression), HR 54 bpm, "
        "BP 88/50 mmHg.",
        "BENZODIAZEPINE_OVERDOSE_AND_TOXICITY",
    ),
    (
        "Acute myocardial infarction",
        "PATIENT ADMISSION NOTE: 55-year-old female presenting with severe retrosternal chest pain radiating "
        "to left jaw and shoulder, with diaphoretic sweating. BP 140/90 mmHg, HR 105 bpm. "
        "Trop-I elevated at 0.85 ng/mL.",
        "ACUTE_CORONARY_SYNDROME_AND_MI",
    ),
    (
        "Organophosphate toxicity",
        "EMERGENCY ROOM ASSESSMENT: 28-year-old agricultural worker with accidental pesticide exposure. "
        "Pinpoint pupils (miosis), profuse salivation, vomiting, wheezing. HR 48 bpm, BP 85/55 mmHg.",
        "ORGANOPHOSPHATE_POISONING_TOXICOLOGY",
    ),
    (
        "Septic shock",
        "EMERGENCY ADMISSION: 71-year-old female with high fever (39.4 C), confused and lethargic. "
        "HR 128, BP 82/48. Lactate 4.8 mmol/L, WBC 22,000 /uL.",
        "SEPSIS_AND_SEPTIC_SHOCK",
    ),
    (
        "Unrelated case (expect Not Sure)",
        "fractured ankle after a fall, swollen, no other symptoms",
        None,
    ),
]


def main() -> int:
    print(f"Knowledge base: {ensure_knowledge_base()} guidelines\n")
    failures = 0
    for name, text, expected in CASES:
        report = diagnose_patient(document_text=text)
        diag = report.primary_diagnosis
        if expected:
            ok = report.reference_guidelines[:1] == [expected]
        else:
            ok = report.source == "no_match" and diag.triage_level.value == "UNDETERMINED"
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       source={report.source} | {diag.condition_name} | {diag.triage_level.value}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())