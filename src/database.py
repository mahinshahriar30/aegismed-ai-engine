import os
import chromadb
from chromadb.utils import embedding_functions

# Set path for local ChromaDB vector store
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medical_reference.txt")

client = chromadb.PersistentClient(path=DB_PATH)

# SentenceTransformer embedding function handles model loading cleanly
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_or_create_collection():
    """Returns or creates the AegisMed medical vector collection."""
    return client.get_or_create_collection(
        name="aegismed_kb",
        embedding_function=embedding_func
    )

def initialize_medical_kb():
    """
    Parses data/medical_reference.txt and populates ChromaDB if empty.
    """
    collection = get_or_create_collection()
    
    # Avoid re-indexing if collection is already populated
    if collection.count() > 0:
        print(f"📦 Vector Store active ({collection.count()} reference guidelines indexed).")
        return

    if not os.path.exists(DATA_FILE):
        print(f"⚠️ Warning: Reference file {DATA_FILE} not found. Skipping initialization.")
        return

    print("🔄 Generating vector embeddings & indexing guidelines into ChromaDB...")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("[DIAGNOSTIC_REF:") if b.strip()]
    
    documents = []
    metadatas = []
    ids = []

    for idx, block in enumerate(blocks):
        full_text = f"[DIAGNOSTIC_REF:{block}"
        documents.append(full_text)
        
        tag_line = block.split("]")[0] if "]" in block else f"REF_{idx}"
        metadatas.append({"ref_tag": tag_line})
        ids.append(f"doc_{idx}")

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Successfully indexed {len(documents)} clinical reference guidelines!")

def query_medical_kb(query_text: str, n_results: int = 3) -> str:
    """
    Performs vector similarity search against ChromaDB.
    """
    collection = get_or_create_collection()
    
    if collection.count() == 0:
        initialize_medical_kb()

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_results, max(1, collection.count()))
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No specific reference guideline found in vector database."

    return "\n\n".join(docs)