# src/database.py

import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="medical_knowledge_base",
    embedding_function=embedding_func
)

def query_medical_kb(query_text: str, n_results: int = 3) -> str:
    """Standard RAG retriever that returns concatenated guideline snippets."""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    if results and results.get("documents"):
        return "\n---\n".join(results["documents"][0])
    return "No matching clinical guidelines found."


def search_medical_kb_direct(query_text: str, similarity_threshold: float = 0.90):
    """
    Queries ChromaDB and returns structured metadata if vector distance 
    indicates a high-confidence direct match.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=1,
        include=["metadatas", "distances"]
    )
    
    if results and results.get('distances') and len(results['distances'][0]) > 0:
        distance = results['distances'][0][0]
        # Cosine/Euclidean distance check (< 0.10 roughly equates to > 90% similarity)
        if distance < (1.0 - similarity_threshold):
            metadata = results['metadatas'][0][0]
            
            # Format actions list cleanly from metadata string
            actions_raw = metadata.get("recommended_actions", "")
            if isinstance(actions_raw, str):
                actions = [a.strip() for a in actions_raw.split(";") if a.strip()]
            else:
                actions = actions_raw

            return {
                "triage_level": metadata.get("triage_level", "Urgent"),
                "condition_name": metadata.get("condition_name", "Identified Condition"),
                "actions": actions
            }
            
    return None