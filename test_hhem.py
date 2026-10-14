"""
Factual-consistency check with Vectara's open-source HHEM-2.1-Open model (Hugging Face:
vectara/hallucination_evaluation_model). It runs on your own computer: no API key, no quota.

For every test case it checks, sentence by sentence, whether the AI's justification and action
steps are SUPPORTED by the matched guideline text plus the patient note. Each pair gets a score from
0 (not supported) to 1 (fully supported). Vectara recommends 0.5 as the cut-off.

Setup (development only, do not add to requirements.txt):
    python -m pip install torch transformers
Run from the project root:
    python test_hhem.py

It reuses the engine answers that test_eval.py saved in eval_engine_cache.json. A case that is not
saved yet is answered by the engine now (needs GROQ_API_KEY, paced for Groq's limit) and saved.

Optional settings (environment variables):
    HHEM_THRESHOLD   support cut-off (default 0.5)
    MAX_CASES        only check the first N cases (default 0 = all)
    ENGINE_PAUSE     seconds between NEW engine calls (default 16)

Security note: loading this model uses trust_remote_code=True, which runs code from the model's
repository. That repository is Vectara's official one, but only do this for sources you trust.
Use made-up (synthetic) notes only.
"""
import hashlib
import json
import os
import re
import sys
import time
from statistics import mean

from src.database import ensure_knowledge_base, retrieve
from src.engine import diagnose_patient
from src.schema import AegisMedAuditResponse

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "test_cases.json")
ENGINE_CACHE_PATH = os.path.join(HERE, "eval_engine_cache.json")
RESULTS_PATH = os.path.join(HERE, "hhem_results.json")

THRESHOLD = float(os.getenv("HHEM_THRESHOLD", "0.5"))
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
ENGINE_PAUSE = float(os.getenv("ENGINE_PAUSE", "16"))
MODEL_ID = "vectara/hallucination_evaluation_model"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def split_sentences(text):
    """Splits a paragraph into sentences, so each claim is checked on its own."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def get_answer(case, cache):
    """The engine's answer for this case: from the saved file, or asked now (and saved)."""
    key = case["id"] + "|" + hashlib.sha256(case["presentation"].encode("utf-8")).hexdigest()[:16]
    if key in cache:
        return AegisMedAuditResponse.model_validate(cache[key]), True
    if not os.getenv("GROQ_API_KEY"):
        return None, False
    audit = None
    for _ in range(4):
        audit = diagnose_patient(case["presentation"])
        if audit.source != "database_fallback":
            break
        print("    AI unavailable (probably Groq's per-minute limit). Waiting 30s, then retrying...", flush=True)
        time.sleep(30)
    if audit.source == "ai_grounded":
        cache[key] = audit.model_dump(mode="json")
        save_json(ENGINE_CACHE_PATH, cache)
    time.sleep(ENGINE_PAUSE)
    return audit, False


def score_pairs(model, pairs, batch_size=8):
    scores = []
    for i in range(0, len(pairs), batch_size):
        out = model.predict(pairs[i:i + batch_size])
        scores.extend(float(s) for s in out.tolist())
    return scores


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        sys.exit("Install the checker first:  python -m pip install torch transformers")

    ensure_knowledge_base()
    cases = load_json(CASES_PATH, [])
    total_in_file = len(cases)
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]
    print(f"Found {total_in_file} test cases in test_cases.json; checking {len(cases)}.")
    cache = load_json(ENGINE_CACHE_PATH, {})

    print(f"Loading {MODEL_ID} (the first run downloads the model)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)

    rows, all_scores, skipped = [], [], []
    for n, case in enumerate(cases, start=1):
        print(f"[{n}/{len(cases)}] {case['id']}", flush=True)
        try:
            audit, from_cache = get_answer(case, cache)
            if audit is None or audit.source != "ai_grounded":
                why = "no saved answer and no GROQ_API_KEY" if audit is None else f"source={audit.source}"
                print(f"    skipped ({why})", flush=True)
                skipped.append(case["id"])
                continue
            context = [h.document for h in retrieve(case["presentation"], 3) if h.matched]
            # Premise = the guideline text the AI was given + the patient note (the AI may restate the note).
            premise = "\n\n".join(context) + "\n\nPatient presentation: " + case["presentation"]
            d = audit.primary_diagnosis
            claims = split_sentences(d.clinical_justification) + [a.strip() for a in audit.immediate_emergency_actions if a.strip()]
            scores = score_pairs(model, [(premise, c) for c in claims])
            row = {
                "id": case["id"],
                "claims": len(claims),
                "mean_score": round(mean(scores), 3),
                "min_score": round(min(scores), 3),
                "all_supported": all(s >= THRESHOLD for s in scores),
                "unsupported": [{"score": round(s, 3), "text": c} for c, s in zip(claims, scores) if s < THRESHOLD],
            }
            rows.append(row)
            all_scores.extend(scores)
            print(f"    {len(claims)} claims | mean {row['mean_score']} | lowest {row['min_score']} | "
                  f"{'all supported' if row['all_supported'] else str(len(row['unsupported'])) + ' below ' + str(THRESHOLD)}", flush=True)
        except Exception as e:
            print(f"    ERROR: {str(e)[:200]}", flush=True)
            skipped.append(case["id"])

    print("\n=== SUMMARY (HHEM-2.1-Open, threshold %.2f) ===" % THRESHOLD)
    print("Test cases in file:", total_in_file, "| checked:", len(rows), "| skipped:", len(skipped), skipped or "")
    if rows:
        supported = sum(1 for s in all_scores if s >= THRESHOLD)
        print("Claims supported by the guideline + note:", pct(supported, len(all_scores)))
        print("Cases where EVERY claim is supported:", pct(sum(r["all_supported"] for r in rows), len(rows)))
        print(f"Mean support score across all claims: {mean(all_scores):.2f}")
        weakest = sorted((u["score"], r["id"], u["text"]) for r in rows for u in r["unsupported"])[:5]
        if weakest:
            print("\nLowest-scoring claims (read these yourself, a low score can also be a harmless paraphrase):")
            for score, cid, text in weakest:
                print(f"  {score:.2f}  {cid}: {text[:140]}")
    save_json(RESULTS_PATH, {"threshold": THRESHOLD, "model": MODEL_ID, "cases": rows, "skipped": skipped})
    print("\nPer-case results saved to hhem_results.json")