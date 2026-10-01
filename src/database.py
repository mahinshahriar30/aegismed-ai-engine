# src/database.py
import logging
import math
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
# A guideline also counts as a match when enough distinctive query words appear in its
# diagnostic criteria (keyword check, backs up the weak embedding model).
LEX_MIN_TERMS = int(os.getenv("LEX_MIN_TERMS", "3"))
LEX_MIN_SCORE = float(os.getenv("LEX_MIN_SCORE", "7.0"))

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
    lex_score: float = 0.0   # weighted keyword overlap with the guideline's criteria
    lex_terms: int = 0       # number of distinct query words found in the criteria
    matched: bool = False    # True = confident enough to treat as a real match

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


_STOP = set("""
a about above after again all also an and any are as at be been before being between both but by can did do
does during each for from had has have having he her here his how into is it its just may might more most no
nor not of off on once only or other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where which while
who will with would you your patient patients presenting presented presents presentation admission admitted
brought note notes year years old male female man woman boy girl found noted mmhg bpm mmol per min hour hours
day days room air
""".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _tokens(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) >= 3 and w not in _STOP}


def _doc_search_text(doc_id: str, document: str) -> str:
    """The symptom-style part of a guideline: name, diagnosis, diagnostic criteria."""
    h = Hit(id=doc_id, document=document)
    parts = [h.title.replace("_", " "), h.condition_name, h._doc_field("Diagnostic Criteria")]
    return ". ".join(p for p in parts if p) or document[:600]


def _build_lex(docs: dict):
    """docs: {id: document text}. Returns (token sets per doc, idf weight per word)."""
    toks = {i: _tokens(_doc_search_text(i, d)) for i, d in docs.items()}
    n = len(toks)
    df: dict = {}
    for t in toks.values():
        for w in t:
            df[w] = df.get(w, 0) + 1
    return toks, {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}


def _lex_scores(query: str, index) -> dict:
    toks, idf = index
    q = _tokens(query)
    out = {}
    for i, t in toks.items():
        common = q & t
        out[i] = (sum(idf[w] for w in common), len(common))
    return out


_lex_cache: dict = {"count": -1, "index": None}


def _lex_index():
    count = collection.count()
    if _lex_cache["count"] != count:
        data = collection.get(include=["documents"])
        docs = {i: (d or "") for i, d in zip(data["ids"], data["documents"])}
        _lex_cache.update(count=count, index=_build_lex(docs))
    return _lex_cache["index"]


def retrieve(query_text: str, n_results: int = 3) -> List[Hit]:
    """
    Best guidelines for the query, matched ones first. Combines the vector search
    with a keyword check. May raise if the database is broken.
    """
    total = collection.count()
    if total == 0:
        return []
    r = collection.query(
        query_texts=[query_text],
        n_results=min(total, 100),
        include=["documents", "metadatas", "distances"],
    )
    ids = (r.get("ids") or [[]])[0]
    docs = (r.get("documents") or [[]])[0]
    metas = (r.get("metadatas") or [[]])[0]
    dists = (r.get("distances") or [[]])[0]

    hits = [
        Hit(
            id=str(_id),
            document=docs[i] if i < len(docs) and docs[i] else "",
            metadata=(metas[i] if i < len(metas) and metas[i] else {}),
            distance=float(dists[i]) if i < len(dists) else 999.0,
        )
        for i, _id in enumerate(ids)
    ]  # already ordered by vector distance

    lex = _lex_scores(query_text, _lex_index())
    for h in hits:
        h.lex_score, h.lex_terms = lex.get(h.id, (0.0, 0))
        h.matched = h.distance <= MAX_MATCH_DISTANCE or (
            h.lex_terms >= LEX_MIN_TERMS and h.lex_score >= LEX_MIN_SCORE
        )

    vec_rank = {h.id: i for i, h in enumerate(hits)}
    lex_rank = {h.id: i for i, h in enumerate(sorted(hits, key=lambda h: -h.lex_score))}
    fused = lambda h: 1.0 / (10 + vec_rank[h.id]) + 1.0 / (10 + lex_rank[h.id])
    hits.sort(key=lambda h: (not h.matched, -fused(h)))
    return hits[:n_results]


# ---------------------------------------------------------------------------
# Knowledge-base loader: builds the collection from data/medical_reference.txt
# ---------------------------------------------------------------------------
DATA_FILE = os.getenv(
    "MEDICAL_DATA_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "medical_reference.txt"),
)


# Bump this whenever the way chunks are embedded changes, so existing databases get rebuilt.
EMBED_VERSION = "criteria-1"


def parse_guideline_blocks(text: str) -> List[dict]:
    """One chunk per '[DIAGNOSTIC_REF: NAME]' block, so each chunk holds exactly one guideline."""
    blocks = []
    for part in re.split(r"(?m)^(?=\[DIAGNOSTIC_REF:)", text):
        m = re.match(r"\[DIAGNOSTIC_REF:\s*([^\]]+?)\s*\]", part)
        if not m:
            continue  # title banner or separator lines
        ref = m.group(1)
        body = re.sub(r"\n=+\s*$", "", part.strip()).strip()
        hit = Hit(id=ref, document=body)
        metadata = {"source": "medical_reference.txt", "domain": "medical", "reference_id": ref}
        if hit.condition_name:
            metadata["condition_name"] = hit.condition_name
        if hit.triage_raw:
            metadata["triage_level"] = hit.triage_raw
        metadata["embed_v"] = EMBED_VERSION
        # Search vectors are built from the symptom-style part only (name, diagnosis,
        # diagnostic criteria), NOT from the long treatment protocol with its drug doses.
        # The full guideline is still what is stored and sent to the LLM.
        embed_text = ". ".join(
            part for part in (
                ref.replace("_", " ").title(),
                hit.condition_name,
                hit._doc_field("Diagnostic Criteria"),
            ) if part
        )
        blocks.append({"id": f"medical_{ref}", "document": body, "metadata": metadata, "embed_text": embed_text})
    return blocks


def ensure_knowledge_base() -> int:
    """
    Makes the collection match the guideline file. Does nothing if it already matches,
    so it is cheap to call on every startup. Returns the number of guidelines stored.
    """
    if not os.path.exists(DATA_FILE):
        logger.error("Guideline file not found: %s", DATA_FILE)
        return collection.count()
    with open(DATA_FILE, encoding="utf-8") as f:
        blocks = parse_guideline_blocks(f.read())
    if not blocks:
        logger.error("No [DIAGNOSTIC_REF: ...] entries found in %s", DATA_FILE)
        return collection.count()

    wanted = {b["id"]: b["document"] for b in blocks}
    current = collection.get(ids=list(wanted), include=["documents", "metadatas"])
    same = dict(zip(current["ids"], current["documents"])) == wanted and all(
        (m or {}).get("embed_v") == EMBED_VERSION for m in current["metadatas"]
    )
    stale = set(collection.get(include=[])["ids"]) - set(wanted)
    if same and not stale:
        return collection.count()

    vectors = embedding_func([b["embed_text"] for b in blocks])
    collection.upsert(
        ids=[b["id"] for b in blocks],
        documents=[b["document"] for b in blocks],
        embeddings=[v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors],
        metadatas=[b["metadata"] for b in blocks],
    )
    if stale:
        collection.delete(ids=list(stale))
    _lex_cache["count"] = -1  # force the keyword index to rebuild
    logger.info("Knowledge base loaded: %d guidelines", collection.count())
    return collection.count()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--ingest"]:
        # Local use:  python -m src.database --ingest
        print(f"Knowledge base now holds {ensure_knowledge_base()} guidelines")
        sys.exit(0)
    # Threshold tuning helper:  python -m src.database "62-year-old male with facial droop..."
    text = " ".join(sys.argv[1:]) or "sudden facial drooping and slurred speech"
    print(f"Collection size: {collection.count()} | MAX_MATCH_DISTANCE={MAX_MATCH_DISTANCE}")
    for h in retrieve(text, 5):
        verdict = "MATCH" if h.matched else "no match"
        print(
            f"{h.distance:7.3f}  keywords={h.lex_score:5.1f}/{h.lex_terms}  {verdict:8}  "
            f"{h.title}  | triage={h.triage_raw or '-'}"
        )