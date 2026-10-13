"""
Batch evaluation of the AegisMed engine with DeepEval, using Gemini as the judge.
If you are logged in to Confident AI, the results are also uploaded to its dashboard.

Setup (development only, do not add to requirements.txt):
    python -m pip install -U deepeval
    deepeval login            (one time: paste your Confident AI key when it asks)
Set BOTH keys in your terminal first (the engine uses Groq, the judge uses Gemini):
    PowerShell:  $env:GROQ_API_KEY="..."
                 $env:GEMINI_API_KEY="..."
Run from the project root:
    python test_eval_suite.py

test_cases.json is a list of cases:
    {"id": "stroke_1", "presentation": "...", "expected": "ACUTE_CEREBROVASCULAR_ACCIDENT_STROKE"}
Use "expected": null for out-of-scope cases (they must NOT be answered from a guideline).
Use made-up (synthetic) notes only: Confident AI stores what you upload on its servers.
"""
import json
import os
import sys
import time
from statistics import mean

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from google import genai
from google.genai import types

from src.database import ensure_knowledge_base, retrieve
from src.engine import diagnose_patient

missing = []
if not os.getenv("GROQ_API_KEY"):
    missing.append("GROQ_API_KEY")
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    missing.append("GEMINI_API_KEY")
if missing:
    sys.exit("Set these in your terminal first: " + ", ".join(missing))


class GeminiEvalModel(DeepEvalBaseLLM):
    """The 'judge' model. It always answers in JSON, because DeepEval's metrics expect JSON."""

    def __init__(self, model_name="gemini-flash-latest"):
        self.model_name = model_name
        # 60-second limit per call, so a hanging connection cannot freeze the run.
        self.client = genai.Client(http_options=types.HttpOptions(timeout=60000))
        self.model = self.client

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", temperature=0
                    ),
                )
                return response.text
            except Exception as e:  # rate limit, wrong model name, bad key, network problem...
                last_error = e
                print(f"    judge call failed (attempt {attempt + 1}/3): {str(e)[:200]}", flush=True)
                time.sleep(5 * (attempt + 1))
        raise last_error

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


if __name__ == "__main__":
    ensure_knowledge_base()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases.json")
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset not found at {dataset_path}. Please create test_cases.json.")
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)[:30]

    # ---------- Phase 1: run the engine on every case (fast) ----------
    print(f"Phase 1: running the engine on {len(cases)} cases...\n", flush=True)
    rows, test_cases = [], []
    for i, case in enumerate(cases, start=1):
        row = {"id": case.get("id", str(i)), "expected": case.get("expected"), "correct": False}
        print(f"[{i}/{len(cases)}] {row['id']}: running the engine...", flush=True)
        started = time.time()
        try:
            presentation = case["presentation"]
            audit = diagnose_patient(presentation)
            top = audit.reference_guidelines[:1]
            row["source"] = audit.source
            row["top_guideline"] = top[0] if top else None
            # In-scope: the right guideline must be on top. Out-of-scope: it must not be grounded.
            row["correct"] = (top == [row["expected"]]) if row["expected"] else (not audit.grounded)

            context = [h.document for h in retrieve(presentation, 3) if h.matched]
            if audit.source == "ai_grounded" and context:
                d = audit.primary_diagnosis
                output = (
                    f"Primary Diagnosis: {d.condition_name}\n"
                    f"Triage Level: {d.triage_level.value}\n"
                    f"Clinical Justification: {d.clinical_justification}\n"
                    f"Immediate Actions: {'; '.join(audit.immediate_emergency_actions)}"
                )
                test_cases.append(
                    LLMTestCase(name=row["id"], input=presentation, actual_output=output,
                                retrieval_context=context)
                )
            print(f"    done in {time.time() - started:.0f}s: source={row['source']} "
                  f"top={row['top_guideline']} correct={row['correct']}", flush=True)
        except Exception as e:  # one failing case must not stop the whole run
            row["error"] = str(e)[:300]
            print(f"    ERROR: {row['error']}", flush=True)
        rows.append(row)

    # ---------- Phase 2: judge the AI answers (uploads to Confident AI if logged in) ----------
    judge_scores = {}
    if test_cases:
        print(f"\nPhase 2: judging {len(test_cases)} AI answers with Gemini. "
              "Failed judge calls are printed as they happen.\n", flush=True)
        judge = GeminiEvalModel()
        metrics = [
            FaithfulnessMetric(threshold=0.8, model=judge, include_reason=True, async_mode=False),
            AnswerRelevancyMetric(threshold=0.8, model=judge, include_reason=True, async_mode=False),
        ]
        kwargs = {}
        try:  # run the judge one call at a time, to stay inside rate limits
            from deepeval.evaluate.configs import AsyncConfig
            kwargs["async_config"] = AsyncConfig(run_async=False)
        except Exception:
            pass
        try:
            result = evaluate(test_cases=test_cases, metrics=metrics, **kwargs)
        except TypeError:
            result = evaluate(test_cases=test_cases, metrics=metrics)
        try:  # read the scores back (attribute names can differ between DeepEval versions)
            for tr in result.test_results:
                for md in tr.metrics_data:
                    judge_scores.setdefault(md.name, []).append((md.score, md.success))
                    for r in rows:
                        if r["id"] == tr.name:
                            r.setdefault("judge", {})[md.name] = md.score
        except Exception:
            judge_scores = {}

    # ---------- Summary ----------
    in_scope = [r for r in rows if r["expected"]]
    out_scope = [r for r in rows if not r["expected"]]
    errors = [r for r in rows if "error" in r]
    print("\n=== SUMMARY ===")
    print("Cases run:", len(rows), "| errors:", len(errors))
    print("In-scope cases matched to the right guideline:", pct(sum(r["correct"] for r in in_scope), len(in_scope)))
    print("Out-of-scope cases correctly NOT grounded:", pct(sum(r["correct"] for r in out_scope), len(out_scope)))
    print(f"AI answers judged: {len(test_cases)} | not judged (no AI answer from a guideline, or error): "
          f"{len(rows) - len(test_cases)}")
    for name, vals in judge_scores.items():
        print(f"{name}: mean {mean(s for s, _ in vals if s is not None):.2f}, "
              f"pass {pct(sum(1 for _, ok in vals if ok), len(vals))}")
    print("\nIf you are logged in to Confident AI, look for the results link in the output above.")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("Per-case results saved to eval_results.json")