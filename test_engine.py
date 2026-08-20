import os
from src.engine import diagnose_patient

def run_medical_test():
    print("🚀 Initializing HouseMd AI Engine Medical Test...")

    sample_medical = """
    PATIENT LAB REPORT:
    Fasting Blood Glucose: 142 mg/dL
    HbA1c: 6.8%
    Total Cholesterol: 245 mg/dL
    ALT (Alanine Aminotransferase): 65 U/L
    AST (Aspartate Aminotransferase): 58 U/L
    Notes: Patient reports fatigue and mild right upper quadrant abdominal discomfort.
    """

    print("\n--------------------------------------------------")
    print("🧪 Executing Diagnostic Test: Medical Lab Report")
    print("--------------------------------------------------")

    try:
        # Executes RAG pipeline and returns AegisMedAuditResponse
        report = diagnose_patient(document_text=sample_medical)

        # Access fields nested inside primary_diagnosis
        diag = getattr(report, 'primary_diagnosis', None)
        
        if diag:
            condition = getattr(diag, 'condition_name', 'N/A')
            triage = getattr(diag, 'triage_level', 'N/A')
            triage_val = getattr(triage, 'value', triage)
            justification = getattr(diag, 'clinical_justification', '')
            ref = getattr(diag, 'reference_guideline', 'N/A')

            print(f"📄 Primary Diagnosis : {condition}")
            print(f"🚨 Triage Severity  : {triage_val}")
            print(f"📚 Ref Guideline    : {ref}")
            print(f"\n📝 Clinical Justification:\n{justification}\n")

        # Handle detected risks if present
        risks = getattr(report, 'detected_risks', None) or getattr(report, 'anomalies', None)
        if risks:
            print("⚠️ Detected Health Risks & Anomalies:")
            for risk in risks:
                severity = getattr(risk, 'severity', 'HIGH')
                issue = getattr(risk, 'issue', getattr(risk, 'risk_name', 'Anomaly'))
                explanation = getattr(risk, 'explanation', '')
                print(f"  - [{severity}] {issue}: {explanation}")

        # Handle recommendations if present
        recs = getattr(report, 'actionable_recommendations', None) or getattr(report, 'recommendations', None)
        if recs:
            print("\n💡 Actionable Recommendations:")
            for rec in recs:
                print(f"  • {rec}")

        print("\n✅ Medical Test Executed Successfully!")

    except Exception as e:
        print(f"❌ Test Failed with error: {str(e)}")

if __name__ == "__main__":
    run_medical_test()