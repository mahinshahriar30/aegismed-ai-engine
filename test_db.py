import os
from src.database import DocuShieldKB

def main():
    print("🚀 Initializing DocuShield Vector Store...")
    kb = DocuShieldKB(db_path="./vector_store", collection_name="docushield_kb")

    # 1. Index our reference files
    legal_path = os.path.join("data", "legal_reference.txt")
    medical_path = os.path.join("data", "medical_reference.txt")

    kb.index_reference_file(legal_path, domain="legal")
    kb.index_reference_file(medical_path, domain="medical")

    print("\n--- 🧪 Test Query 1: Legal Domain Search ---")
    query_legal = "What is the rule about monthly non-refundable cleaning fees and § 402?"
    results_legal = kb.hybrid_search(query=query_legal, domain_filter="legal", top_k=2)
    for i, res in enumerate(results_legal, 1):
        print(f"Result {i}:\n{res}\n")

    print("--- 🧪 Test Query 2: Medical Domain Search ---")
    query_medical = "What does an HbA1c level of 6.2% mean?"
    results_medical = kb.hybrid_search(query=query_medical, domain_filter="medical", top_k=2)
    for i, res in enumerate(results_medical, 1):
        print(f"Result {i}:\n{res}\n")

if __name__ == "__main__":
    main()