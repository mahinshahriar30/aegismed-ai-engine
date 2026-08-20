import os
import requests
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="HouseMd AI - Emergency Clinical Decision Support",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------------------------------------------------
# PROMINENT DISCLAIMER BANNER
# -----------------------------------------------------------------------------
st.title("🏥 HouseMd AI Engine")
st.caption("Proof-of-Concept Emergency Triage & Diagnostic Decision Support Engine")

st.error("""
🚨 **IMPORTANT NOTICE FOR CLINICIANS & TESTERS:**
* **RESEARCH PROTOTYPE ONLY:** This application is an experimental proof-of-concept created solely for software benchmarking, research evaluation, and technical demonstrations.
* **NOT A MEDICAL DEVICE:** This system does **NOT** provide medical advice, diagnosis, or treatment plans. It is not a substitute for professional clinical judgment or established hospital triage protocols.
* **DO NOT USE FOR ACTIVE PATIENT CARE:** Do not enter real Protected Health Information (PHI) or rely on this system during active clinical emergencies.
""")

st.markdown("---")

# -----------------------------------------------------------------------------
# BACKEND URL CONFIGURATION
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# SIDEBAR - PRE-LOADED CLINICAL SCENARIOS
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Sample Clinical Scenarios")
sample_choice = st.sidebar.selectbox(
    "Select a pre-loaded test case to benchmark:",
    [
        "Custom Input",
        "Acute Stroke (CVA)",
        "Benzodiazepine Overdose",
        "Acute Myocardial Infarction",
        "Organophosphate Toxicity",
        "Septic Shock"
    ]
)

SAMPLES = {
    "Acute Stroke (CVA)": "PATIENT EMERGENCY ADMISSION: 62-year-old male presenting with sudden right-sided facial drooping, slurred speech, and weakness in the right arm. BP: 195/110 mmHg. Onset: 90 minutes ago.",
    "Benzodiazepine Overdose": "EMERGENCY ADMISSION: 29-year-old female brought in unresponsive next to empty pill bottles of Alprazolam (Xanax). SpO2 86% on room air, RR 8/min (severe respiratory depression), HR 54 bpm, BP 88/50 mmHg.",
    "Acute Myocardial Infarction": "PATIENT ADMISSION NOTE: 55-year-old female presenting with severe retrosternal chest pain radiating to left jaw and shoulder, with diaphoretic sweating. BP 140/90 mmHg, HR 105 bpm. Trop-I elevated at 0.85 ng/mL.",
    "Organophosphate Toxicity": "EMERGENCY ROOM ASSESSMENT: 28-year-old agricultural worker with accidental pesticide exposure. Pinpoint pupils (miosis), profuse salivation, vomiting, wheezing. HR 48 bpm, BP 85/55 mmHg.",
    "Septic Shock": "EMERGENCY ADMISSION: 71-year-old female with high fever (39.4°C), confused and lethargic. HR 128, BP 82/48 mmHg. Lactate 4.8 mmol/L, WBC 22,000 /uL."
}

default_text = SAMPLES.get(sample_choice, "") if sample_choice != "Custom Input" else ""

# -----------------------------------------------------------------------------
# MAIN INPUT SECTION
# -----------------------------------------------------------------------------
document_text = st.text_area(
    "Paste Patient Presentation / Clinical Notes:",
    value=default_text,
    height=180,
    placeholder="Enter patient symptoms, vitals, lab values, or emergency presentation notes here..."
)

if st.button("🚀 Analyze & Generate Clinical Audit", type="primary"):
    if not document_text.strip():
        st.error("Please enter patient clinical notes before submitting.")
    else:
        with st.spinner("Querying ChromaDB vector database & running Gemini evaluation..."):
            try:
                # Call FastAPI Backend
                response = requests.post(
                    f"{BACKEND_URL}/api/v1/diagnose",
                    json={"document_text": document_text, "domain": "medical"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("Clinical Analysis Generated Successfully!")
                    st.subheader("📊 Structured Triage & Diagnostic Output")
                    
                    # Extract Primary Diagnosis Object
                    primary_diag = data.get("primary_diagnosis", {})
                    if isinstance(primary_diag, str):
                        condition_name = primary_diag
                        triage_level = data.get("triage_level", "UNKNOWN")
                        justification = data.get("clinical_justification", "N/A")
                        ref_guideline = data.get("reference_guideline", "N/A")
                    else:
                        condition_name = primary_diag.get("condition_name", "N/A")
                        triage_level = primary_diag.get("triage_level", "UNKNOWN")
                        justification = primary_diag.get("clinical_justification", "N/A")
                        ref_guideline = primary_diag.get("reference_guideline", "N/A")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if triage_level == "CRITICAL_EMERGENCY":
                            st.error(f"🚨 **TRIAGE LEVEL:** {triage_level}")
                        elif triage_level == "HIGH_PRIORITY":
                            st.warning(f"⚠️ **TRIAGE LEVEL:** {triage_level}")
                        else:
                            st.success(f"🟢 **TRIAGE LEVEL:** {triage_level}")
                    
                    with col2:
                        st.markdown(f"### Primary Diagnosis:\n**{condition_name}**")

                    st.markdown("---")
                    
                    # Patient Summary
                    st.markdown("### 🔍 Patient Presentation Summary")
                    st.write(data.get("patient_summary", "N/A"))

                    # Immediate Emergency Actions
                    st.markdown("### ⚡ Recommended Immediate Emergency Actions")
                    actions = data.get("immediate_emergency_actions", [])
                    if actions:
                        for action in actions:
                            st.markdown(f"- {action}")
                    else:
                        st.write("No immediate emergency actions specified.")

                    # Differential Diagnoses (if present)
                    diff_diagnoses = data.get("differential_diagnoses", [])
                    if diff_diagnoses:
                        st.markdown("### 🩺 Differential Diagnoses")
                        for diff in diff_diagnoses:
                            st.markdown(f"- **{diff.get('condition_name')}** (*{diff.get('triage_level')}*): {diff.get('clinical_justification')}")

                    # Clinical Justification & Guideline
                    st.markdown("### 📜 RAG Clinical Justification & Guideline")
                    st.info(f"**Clinical Justification:** {justification}\n\n**Reference Guideline:** `{ref_guideline}`")

                else:
                    st.error(f"Backend API Error (Status {response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to AegisMed API at `{BACKEND_URL}`. Error details: {str(e)}")