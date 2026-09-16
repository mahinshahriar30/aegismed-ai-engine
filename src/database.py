import os
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai
from google.genai import types

# Set path for local ChromaDB vector store
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medical_reference.txt")

client = chromadb.PersistentClient(path=DB_PATH)

class GoogleGenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Wrapper to connect ChromaDB to Google GenAI embedding API."""
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for embeddings.")
        # Target v1 API version to avoid v1beta model resolution errors
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(api_version="v1")
        )
        self.model_name = model_name

    def name(self) -> str:
        return f"google_genai_{self.model_name.replace('-', '_')}"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        batch_size = 16  # Chunk into safe batch sizes
        input_list = list(input)
        
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            for e in response.embeddings:
                embeddings.append(e.values)
                    
        return embeddings

# Initialize embedding function
embedding_func = GoogleGenAIEmbeddingFunction()

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