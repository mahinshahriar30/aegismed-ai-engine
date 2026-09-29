# src/database.py
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger("aegismed.database")

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# A guideline only counts as a "match" if its vector distance is <= this value.
# Smaller = stricter. Your collection uses Chroma's default metric (squared L2), where
# 0 is identical and larger is less similar. 0.9 is a STARTING POINT: tune it by running
#   python -m src.database "your sample case text"
# on cases that should match and cases that should not.
MAX_MATCH_DISTANCE = float(os.getenv("MAX_MATCH_DISTANCE", "0.9"))

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="medical_knowledge_base",
    embedding_function=embedding_func,
)


def _first(metadata: dict, keys) -> str:
    for k in keys:
        v = metadata.get(k)
        if v:
            return str(v).strip()
    return ""


@dataclass
class Hit:
    id: str
    document: str
    metadata: dict = field(default_factory=dict)
    distance: float = 999.0

    def _doc_field(self, label: str) -> str:
        """
        Reads a '- Label: value' line from the guideline text. Only trusted when this
        chunk holds exactly ONE '[DIAGNOSTIC_REF: ...]' entry, so we can never read the
        urgency of a different guideline by mistake. Otherwise returns "".
        """
        doc = self.document or ""
        if len(re.findall(r"\[DIAGNOSTIC_REF:", doc)) != 1:
            return ""
        m = re.search(rf"^\s*-?\s*{label}:\s*(.+?)\s*$", doc, re.MULTILINE | re.IGNORECASE)
        return m.group(1).rstrip(".").strip() if m else ""

    @property
    def condition_name(self) -> str:
        return (
            _first(self.metadata, ("condition_name", "primary_condition", "condition", "name"))
            or self._doc_field("Primary Diagnosis")
        )

    @property
    def triage_raw(self) -> str:
        return _first(self.metadata, ("triage_level", "triage", "urgency")) or self._doc_field("Urgency")

    @property
    def title(self) -> str:
        """Human-readable guideline label, used as the reference shown to the user."""
        m = re.search(r"\[DIAGNOSTIC_REF:\s*([^\]]+?)\s*\]", self.document or "")
        if m:
            return m.group(1)
        if self.condition_name:
            return self.condition_name
        first_line = (self.document or "").strip().splitlines()[0:1]
        return (first_line[0][:80] if first_line else self.id) or self.id

    @property
    def actions(self) -> List[str]:
        raw = (
            self.metadata.get("immediate_emergency_actions")
            or self.metadata.get("recommended_actions")
            or self.metadata.get("actions")
            or ""
        )
        if isinstance(raw, list):
            return [str(a).strip() for a in raw if str(a).strip()]
        from_meta = [a.strip() for a in str(raw).split(";") if a.strip()]
        if from_meta:
            return from_meta
        if len(re.findall(r"\[DIAGNOSTIC_REF:", self.document or "")) == 1:
            return re.findall(r"^\s*\d+\.\s+(.+?)\s*$", self.document or "", re.MULTILINE)
        return []


def retrieve(query_text: str, n_results: int = 3) -> List[Hit]:
    """Nearest guideline chunks WITH distances. May raise if the database is broken."""
    r = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    ids = (r.get("ids") or [[]])[0]
    docs = (r.get("documents") or [[]])[0]
    metas = (r.get("metadatas") or [[]])[0]
    dists = (r.get("distances") or [[]])[0]

    hits = []
    for i, _id in enumerate(ids):
        hits.append(
            Hit(
                id=str(_id),
                document=docs[i] if i < len(docs) and docs[i] else "",
                metadata=(metas[i] if i < len(metas) and metas[i] else {}),
                distance=float(dists[i]) if i < len(dists) else 999.0,
            )
        )
    return hits


# --- Backward-compatible helpers (in case test_db.py or other scripts import them) ---
def query_medical_kb(query_text: str, n_results: int = 3) -> str:
    hits = retrieve(query_text, n_results)
    if not hits:
        return "No matching clinical guidelines found."
    return "\n---\n".join(h.document for h in hits)


def search_medical_kb_direct(query_text: str, similarity_threshold: float = 0.90):
    hits = retrieve(query_text, 1)
    if hits and hits[0].distance <= MAX_MATCH_DISTANCE:
        h = hits[0]
        return {
            "triage_level": h.triage_raw or "UNDETERMINED",
            "condition_name": h.condition_name or h.title,
            "actions": h.actions,
        }
    return None


if __name__ == "__main__":
    # Threshold tuning helper:  python -m src.database "62-year-old male with facial droop..."
    text = " ".join(sys.argv[1:]) or "sudden facial drooping and slurred speech"
    print(f"Collection size: {collection.count()} | MAX_MATCH_DISTANCE={MAX_MATCH_DISTANCE}")
    for h in retrieve(text, 5):
        verdict = "MATCH" if h.distance <= MAX_MATCH_DISTANCE else "no match"
        print(f"{h.distance:7.3f}  {verdict:8}  {h.title}  | triage={h.triage_raw or '-'}")