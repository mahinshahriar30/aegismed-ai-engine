"""
Evaluates the AegisMed engine on EVERY case in test_cases.json with DeepEval, and uploads the
results to your Confident AI dashboard (run `deepeval login` once first).

Built for free-tier rate limits:
  * Every engine answer and every judge reply is saved to disk (eval_engine_cache.json and
    eval_judge_cache.json). If a limit stops the run, just run the command again: finished
    work is reused and costs no quota, and the run continues where it stopped.
  * Calls are paced, and rate-limit errors wait for the time the provider asks for.
  * If the daily quota is used up, the script stops cleanly BEFORE uploading, so the dashboard
    never receives a half-empty run. Run it again after the quota resets.

Setup (development only, do not add to requirements.txt):
    python -m pip install -U deepeval "portalocker[win32]"
    deepeval login
In your terminal, set the keys first (never write them inside a file):
    PowerShell:  $env:GROQ_API_KEY="..."      (used by the engine)
                 $env:GEMINI_API_KEY="..."    (used by the judge, default)
Run from the project root:
    python test_eval.py

Optional settings (environment variables):
    JUDGE               "gemini" (default) or "groq" (then the judge is openai/gpt-oss-120b)
    JUDGE_MODEL         override the judge model name
    JUDGE_MIN_INTERVAL  seconds between judge calls (default 6)
    ENGINE_PAUSE        seconds between engine calls (default 16, for Groq's 8,000 tokens/minute)
    MAX_CASES           only run the first N cases (default 0 = all)
    INCLUDE_REASON      1 (default) adds a written reason per score (one extra call per metric)
    RUN_NAME            name of the run on the dashboard (default AegisMed-eval)

test_cases.json: a list of {"id": ..., "presentation": ...}. "expected" is optional: if missing, the
guideline is taken from the id (see EXPECTED_BY_ID). Use "expected": null for out-of-scope notes.
Use made-up (synthetic) notes only: Confident AI and Gemini's free tier store what you send.
"""
import hashlib
import json
import os
import re
import sys
import time
from statistics import mean

# Give DeepEval more time per case, because paced judge calls are slow (ignored if a version
# does not know these names).
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from google import genai
from google.genai import types
from groq import Groq

from src.database import ensure_knowledge_base, retrieve
from src.engine import diagnose_patient
from src.schema import AegisMedAuditResponse

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_PATH = os.path.join(HERE, "test_cases.json")
ENGINE_CACHE_PATH = os.path.join(HERE, "eval_engine_cache.json")
JUDGE_CACHE_PATH = os.path.join(HERE, "eval_judge_cache.json")
RESULTS_PATH = os.path.join(HERE, "eval_results.json")

JUDGE = os.getenv("JUDGE", "gemini").lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or ("gemini-flash-lite-latest" if JUDGE == "gemini" else "openai/gpt-oss-120b")
JUDGE_MIN_INTERVAL = float(os.getenv("JUDGE_MIN_INTERVAL", "6"))
ENGINE_PAUSE = float(os.getenv("ENGINE_PAUSE", "16"))
MAX_CASES = int(os.getenv("MAX_CASES", "0"))
INCLUDE_REASON = os.getenv("INCLUDE_REASON", "1") == "1"
RUN_NAME = os.getenv("RUN_NAME", "AegisMed-eval")

# The intended guideline of each case, taken from its id (STROKE_01 is meant to be a stroke case).
# A clinician should confirm that every note really describes the condition in its id.
EXPECTED_BY_ID = {
    "STROKE_01": "ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE",
    "ACS_01": "ACUTE_CORONARY_SYNDROME_AND_MI",
    "PE_01": "ACUTE_PULMONARY_EMBOLISM_PE",
    "SEPSIS_01": "SEPSIS_AND_SEPTIC_SHOCK",
    "DKA_01": "DIABETIC_KETOACIDOSIS_DKA_AND_HHS",
    "DHF_01": "DENGUE_HEMORRHAGIC_FEVER_DHF",
    "AKI_01": "ACUTE_RENAL_FAILURE_AKI",
    "PANCREATITIS_01": "ACUTE_PANCREATITIS",
    "ORGANOPHOSPHATE_01": "ORGANOPHOSPHATE_POISONING_TOXICOLOGY",
    "PARACETAMOL_01": "PARACETAMOL_ACETAMINOPHEN_OVERDOSE",
    "SNAKEBITE_01": "SNAKEBITE_ENVENOMATION",
    "BENZODIAZEPINE_01": "BENZODIAZEPINE_OVERDOSE_AND_TOXICITY",
    "ANAPHYLAXIS_01": "ANAPHYLACTIC_SHOCK",
    "TENSION_PNEUMOTHORAX_01": "TENSION_PNEUMOTHORAX",
    "TRAUMA_01": "SEVERE_TRAUMA_AND_HEMORRHAGIC_SHOCK",
    "CO_POISONING_01": "ACUTE_CARBON_MONOXIDE_POISONING",
    "COPD_01": "ACUTE_EXACERBATION_OF_COPD",
    "HEART_FAILURE_01": "ACUTE_DECOMPENSATED_HEART_FAILURE_AND_PULMONARY_EDEMA",
    "HYPERTENSIVE_EMERGENCY_01": "HYPERTENSIVE_EMERGENCY_AND_ENCEPHALOPATHY",
    "CARDIAC_ARREST_01": "CARDIAC_ARREST_ACLS",
    "RTA_01": "ROAD_TRAFFIC_ACCIDENT_RTA_AND_POLYTRAUMA",
    "TBI_01": "TRAUMATIC_BRAIN_INJURY_TBI_AND_EPIDURAL_HEMORRHAGE",
    "DROWNING_01": "DROWNING_AND_NEAR_DROWNING_ASPHYXIA",
    "HEAT_STROKE_01": "HEAT_STROKE_AND_HYPERTHERMIA",
    "GI_BLEED_01": "ACUTE_UPPER_GASTROINTESTINAL_BLEEDING_VARICEAL",
    "HEPATIC_ENCEPHALOPATHY_01": "ACUTE_HEPATIC_ENCEPHALOPATHY_AND_FULMINANT_LIVER_FAILURE",
    "PERFORATED_ULCER_01": "PERFORATED_PEPTIC_ULCER_AND_ACUTE_PERITONITIS",
    "CHOLERA_01": "SEVERE_ACUTE_DIARRHEA_AND_CHOLERA_SHOCK",
    "MALARIA_01": "SEVERE_FALCIPARUM_MALARIA",
    "RABIES_01": "RABIES_ENCEPHALITIS_PREVENTION_AND_EXPOSURE",
}

missing = []
if not os.getenv("GROQ_API_KEY"):
    missing.append("GROQ_API_KEY")
if JUDGE == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    missing.append("GEMINI_API_KEY")
if missing:
    sys.exit("Set these in your terminal first: " + ", ".join(missing))


# ---------------------------------------------------------------- small helpers
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
    os.replace(tmp, path)  # never leaves a half-written cache file


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def retry_delay(message, attempt):
    """How long to wait after a rate-limit error: the provider's own hint if there is one."""
    m = re.search(r"(?:try again|retry) in (?:(\d+)m)?\s*([\d.]+)\s*s", message, re.I)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    m = re.search(r'"retryDelay":\s*"([\d.]+)s"', message)
    if m:
        return float(m.group(1)) + 1
    return min(10 * 2 ** attempt, 60)


class QuotaExhausted(Exception):
    pass


# ---------------------------------------------------------------- the judge
class RateLimitedJudge(DeepEvalBaseLLM):
    """DeepEval's judge model: paced, cached on disk, and patient with rate limits."""

    def __init__(self):
        self.model_name = JUDGE_MODEL
        self.cache = load_json(JUDGE_CACHE_PATH, {})
        self.last_call = 0.0
        self.calls_made = 0
        self.cache_hits = 0
        if JUDGE == "gemini":
            self.client = genai.Client(http_options=types.HttpOptions(timeout=60000))
        else:
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0, max_retries=0)
        self.model = self.client

    def load_model(self):
        return self.client

    def _call(self, prompt):
        if JUDGE == "gemini":
            r = self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            text = r.text
        else:
            r = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=0,
            )
            text = r.choices[0].message.content
        if not text:
            raise ValueError("The judge model returned an empty answer.")
        return text

    def generate(self, prompt: str) -> str:
        key = hashlib.sha256((self.model_name + "\n" + prompt).encode("utf-8")).hexdigest()
        if key in self.cache:  # already paid for in an earlier run
            self.cache_hits += 1
            return self.cache[key]
        for attempt in range(6):
            pause = JUDGE_MIN_INTERVAL - (time.time() - self.last_call)
            if pause > 0:
                time.sleep(pause)
            try:
                self.last_call = time.time()
                text = self._call(prompt)
                self.calls_made += 1
                self.cache[key] = text
                save_json(JUDGE_CACHE_PATH, self.cache)
                return text
            except Exception as e:
                message = str(e)
                if not any(s in message for s in ("429", "RESOURCE_EXHAUSTED", "rate_limit", "Rate limit", "quota")):
                    raise  # a real error (wrong model name, bad key...): stop and show it
                delay = retry_delay(message, attempt)
                if delay > 120:  # the daily quota is gone, waiting here would take too long
                    raise QuotaExhausted(f"Judge quota used up (provider asks to wait {delay / 60:.0f} min): {message[:200]}")
                print(f"    judge rate limit: waiting {delay:.0f}s (attempt {attempt + 1}/6)", flush=True)
                time.sleep(delay)
        raise QuotaExhausted("Judge is still rate-limited after 6 attempts.")

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def check_judge_model():
    """Stops early, with the names that do exist, if the Gemini judge model name is wrong."""
    if JUDGE != "gemini":
        return
    try:
        client = genai.Client()
        names = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception as e:
        print(f"(Could not list Gemini models, continuing anyway: {str(e)[:120]})")
        return
    if JUDGE_MODEL not in names:
        flash = [n for n in names if "flash" in n and "lite" in n] or [n for n in names if "flash" in n]
        sys.exit(f"Judge model '{JUDGE_MODEL}' was not found for your key.\n"
                 f"Flash models available to you: {flash}\n"
                 'Set one with:  $env:JUDGE_MODEL="name-from-the-list"')


# ---------------------------------------------------------------- the engine, with a cache
def run_engine(case, engine_cache):
    """Returns (audit, from_cache). Waits and retries when Groq's per-minute limit hits."""
    key = case["id"] + "|" + hashlib.sha256(case["presentation"].encode("utf-8")).hexdigest()[:16]
    if key in engine_cache:
        return AegisMedAuditResponse.model_validate(engine_cache[key]), True
    audit = None
    for _ in range(4):
        audit = diagnose_patient(case["presentation"])
        if audit.source != "database_fallback":
            break
        print("    AI unavailable (probably Groq's per-minute limit). Waiting 30s, then retrying...", flush=True)
        time.sleep(30)
    if audit.source == "ai_grounded":
        engine_cache[key] = audit.model_dump(mode="json")
        save_json(ENGINE_CACHE_PATH, engine_cache)
    time.sleep(ENGINE_PAUSE)  # stay under Groq's tokens-per-minute limit
    return audit, False


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ensure_knowledge_base()
    check_judge_model()

    if not os.path.exists(CASES_PATH):
        sys.exit(f"Dataset not found at {CASES_PATH}. Please create test_cases.json.")
    cases = load_json(CASES_PATH, [])
    total_in_file = len(cases)
    if MAX_CASES > 0:
        cases = cases[:MAX_CASES]

    for c in cases:
        if "expected" not in c:
            c["expected"] = EXPECTED_BY_ID.get(c.get("id"), "__UNLABELLED__")
    unlabelled = [c.get("id") for c in cases if c["expected"] == "__UNLABELLED__"]
    cases = [c for c in cases if c["expected"] != "__UNLABELLED__"]
    ids = [c["id"] for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})

    print(f"Found {total_in_file} test cases in test_cases.json; running {len(cases)} of them "
          f"({sum(1 for c in cases if c['expected'])} in-scope, {sum(1 for c in cases if not c['expected'])} out-of-scope).")
    if unlabelled:
        print("WARNING: skipped (no 'expected' field and unknown id):", unlabelled)
    if dupes:
        print("WARNING: duplicate ids (cached answers could be mixed up):", dupes)
    if not any(c["expected"] is None for c in cases):
        print("Note: there are no out-of-scope cases, so wrongly matched notes are not measured.")
    print(f"Judge: {JUDGE} / {JUDGE_MODEL}\n")

    # ---------- Phase 1: retrieval accuracy on every case (no AI calls, so no rate limits) ----------
    print("Phase 1: checking which guideline each case is matched to...", flush=True)
    rows = []
    for case in cases:
        row = {"id": case["id"], "expected": case["expected"], "correct": False}
        try:
            matched = [h for h in retrieve(case["presentation"], 3) if h.matched]
            row["top_guideline"] = matched[0].title if matched else None
            row["correct"] = (row["top_guideline"] == row["expected"]) if row["expected"] else (not matched)
        except Exception as e:
            row["error"] = str(e)[:300]
        rows.append(row)
        if not row["correct"]:
            print(f"    MISS {row['id']}: expected={row['expected']} got={row.get('top_guideline')}", flush=True)
    in_scope = [r for r in rows if r["expected"]]
    out_scope = [r for r in rows if not r["expected"]]
    print("  In-scope cases matched to the right guideline:", pct(sum(r["correct"] for r in in_scope), len(in_scope)))
    print("  Out-of-scope cases correctly NOT matched:", pct(sum(r["correct"] for r in out_scope), len(out_scope)))

    # ---------- Phase 2: the engine's AI answer for every correctly matched case ----------
    candidates = [c for c, r in zip(cases, rows) if r["expected"] and r["correct"]]
    engine_cache = load_json(ENGINE_CACHE_PATH, {})
    print(f"\nPhase 2: engine answers for {len(candidates)} matched cases "
          f"(saved answers are reused, new ones are paced)...", flush=True)
    test_cases, ai_unavailable = [], []
    for n, case in enumerate(candidates, start=1):
        print(f"[{n}/{len(candidates)}] {case['id']}", flush=True)
        try:
            audit, cached = run_engine(case, engine_cache)
            print(f"    source={audit.source}{' (from cache)' if cached else ''}", flush=True)
            context = [h.document for h in retrieve(case["presentation"], 3) if h.matched]
            if audit.source == "ai_grounded" and context:
                d = audit.primary_diagnosis
                output = (
                    f"Primary Diagnosis: {d.condition_name}\n"
                    f"Triage Level: {d.triage_level.value}\n"
                    f"Clinical Justification: {d.clinical_justification}\n"
                    f"Immediate Actions: {'; '.join(audit.immediate_emergency_actions)}"
                )
                test_cases.append(LLMTestCase(name=case["id"], input=case["presentation"],
                                              actual_output=output, retrieval_context=context))
            else:
                ai_unavailable.append(case["id"])
        except Exception as e:
            ai_unavailable.append(case["id"])
            print(f"    ERROR: {str(e)[:200]}", flush=True)

    # ---------- Phase 3: judge the AI answers and upload to Confident AI ----------
    judge_scores = {}
    judge = None
    stopped = False
    if test_cases:
        print(f"\nPhase 3: judging {len(test_cases)} AI answers (about {len(test_cases) * (7 if INCLUDE_REASON else 5)} "
              f"judge calls on a first run; saved ones are free)...", flush=True)
        judge = RateLimitedJudge()
        metrics = [
            FaithfulnessMetric(threshold=0.8, model=judge, include_reason=INCLUDE_REASON, async_mode=False),
            AnswerRelevancyMetric(threshold=0.8, model=judge, include_reason=INCLUDE_REASON, async_mode=False),
        ]
        kwargs = {}
        try:  # one judge call at a time
            from deepeval.evaluate.configs import AsyncConfig
            kwargs["async_config"] = AsyncConfig(run_async=False)
        except Exception:
            pass
        try:
            try:
                result = evaluate(test_cases=test_cases, metrics=metrics, identifier=RUN_NAME, **kwargs)
            except TypeError:  # this DeepEval version has no 'identifier' option
                result = evaluate(test_cases=test_cases, metrics=metrics, **kwargs)
            try:  # read the scores back (attribute names can differ between DeepEval versions)
                for tr in result.test_results:
                    for md in tr.metrics_data:
                        if md.score is None:
                            continue
                        judge_scores.setdefault(md.name, []).append((md.score, md.success))
                        for r in rows:
                            if r["id"] == tr.name:
                                r.setdefault("judge", {})[md.name] = md.score
            except Exception:
                judge_scores = {}
        except Exception as e:
            stopped = True
            print("\n" + "=" * 70)
            print("The evaluation stopped before it finished, so NOTHING was uploaded to the dashboard.")
            print("Reason:", str(e)[:400])
            print("Your progress is saved. Run the same command again (after the quota resets if a")
            print("daily limit was hit) and it will continue where it stopped, reusing saved work.")
            print("=" * 70)

    # ---------- Summary ----------
    print("\n=== SUMMARY ===")
    print("Test cases in file:", total_in_file, "| run:", len(cases))
    print("Retrieval, in-scope cases matched to the right guideline:", pct(sum(r["correct"] for r in in_scope), len(in_scope)))
    print("Retrieval, out-of-scope cases correctly NOT matched:", pct(sum(r["correct"] for r in out_scope), len(out_scope)))
    print(f"AI answers: {len(test_cases)} grounded answers out of {len(candidates)} matched cases"
          + (f" | no AI answer for: {ai_unavailable}" if ai_unavailable else ""))
    for name, vals in judge_scores.items():
        print(f"{name}: mean {mean(s for s, _ in vals):.2f}, pass {pct(sum(1 for _, ok in vals if ok), len(vals))}")
    if judge:
        print(f"Judge calls: {judge.calls_made} new, {judge.cache_hits} reused from saved work")
    if stopped:
        print("Status: STOPPED EARLY (see the message above). Run again to continue.")
    elif test_cases:
        print("Status: complete. If you are logged in to Confident AI, look for the results link above.")

    save_json(RESULTS_PATH, rows)
    print("Per-case results saved to eval_results.json")