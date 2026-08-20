import os
from src.engine import DocuShieldEngine

def run_tests():
    print("🚀 Initializing DocuShield AI Engine...")
    engine = DocuShieldEngine()

    # --- Test 1: Legal Lease Addendum ---
    sample_lease = """
    TENANT LEASE ADDENDUM:
    Section 4.1: Landlord reserves the right to enter the premises at any time without prior notice.
    Section 8.2: Security deposit of $2,000 shall be non-refundable under all circumstances, regardless of property condition upon move-out.
    Section 12.0: Tenant is responsible for paying a $500 fee for any maintenance request submitted, including emergency plumbing repairs.
    """

    print("\n--------------------------------------------------")
    print("🧪 Executing Audit Test 1: Legal Lease Addendum")
    print("--------------------------------------------------")
    report_legal = engine.audit_document(document_text=sample_lease, domain="legal")
    print(f"📄 Document Type: {report_legal.document_type} | Domain: {report_legal.domain}")
    print(f"\n📝 Plain Language Translation:\n{report_legal.translated_text}\n")
    print("⚠️ Detected Risks:")
    for risk in report_legal.detected_risks:
        print(f"  - [{risk.severity}] {risk.issue}: {risk.explanation}")
        print(f"    Ref Clause: {risk.reference_clause}")
    print("\n💡 Actionable Recommendations:")
    for rec in report_legal.actionable_recommendations:
        print(f"  • {rec}")

    # --- Test 2: Medical Blood Panel Audit ---
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
    print("🧪 Executing Audit Test 2: Medical Lab Report")
    print("--------------------------------------------------")
    report_med = engine.audit_document(document_text=sample_medical, domain="medical")
    print(f"📄 Document Type: {report_med.document_type} | Domain: {report_med.domain}")
    print(f"\n📝 Plain Language Translation:\n{report_med.translated_text}\n")
    print("⚠️ Detected Health Risks & Anomalies:")
    for risk in report_med.detected_risks:
        print(f"  - [{risk.severity}] {risk.issue}: {risk.explanation}")
        print(f"    Ref Clause: {risk.reference_clause}")
    print("\n💡 Actionable Recommendations:")
    for rec in report_med.actionable_recommendations:
        print(f"  • {rec}")

if __name__ == "__main__":
    run_tests()