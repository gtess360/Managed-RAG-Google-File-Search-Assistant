"""Live evaluator for the Gemini File Search knowledge assistant.

Runs every question in evaluation/questions.json through ``query_pipeline`` and reports:
- status/behavior match (answer vs abstain),
- expected-fact coverage and forbidden-claim violations,
- source recall (expected source files appear in citations),
- latency percentiles.

Requires GEMINI_API_KEY and an ingested store, so it makes real API calls and is
guarded behind the explicit ``--live`` flag.

Usage:
    python -m evaluation.evaluate --live [--split development|holdout] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PATHS, GEMINI_MODEL  # noqa: E402
from src.pipeline import query_pipeline  # noqa: E402

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_questions(split: str | None = None, limit: int | None = None) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if split:
        questions = [q for q in questions if q.get("split") == split]
    return questions[:limit] if limit else questions


def _norm_source(path: str) -> str:
    return Path(path).name.lower()


def evaluate_case(q: dict, result: dict) -> dict:
    answer = (result.get("answer") or "").lower()
    expects_answer = q.get("expected_behavior") == "answer"
    got_answer = result.get("status") in ("answered", "partial", "conflict")

    facts = q.get("expected_answer_facts", [])
    facts_found = [f for f in facts if f.lower() in answer]
    forbidden_hit = [f for f in q.get("forbidden_claims", []) if f.lower() in answer]

    expected_sources = {_norm_source(s) for s in q.get("expected_sources", [])}
    cited = {_norm_source(c.get("source", "")) for c in result.get("citations", [])}
    source_recall = (
        len(expected_sources & cited) / len(expected_sources) if expected_sources else None
    )

    return {
        "id": q["id"],
        "category": q.get("category"),
        "question": q["question"],
        "status": result.get("status"),
        "behavior_match": expects_answer == got_answer and result.get("status") != "error",
        "fact_coverage": (len(facts_found) / len(facts)) if facts else None,
        "forbidden_violations": forbidden_hit,
        "source_recall": source_recall,
        "total_ms": result.get("timings", {}).get("total_ms", 0.0),
        "error": result.get("error"),
        "answer": result.get("answer"),
    }


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


def summarize(cases: list[dict]) -> dict:
    def avg(key):
        vals = [c[key] for c in cases if c[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    lat = [c["total_ms"] for c in cases]
    return {
        "model": GEMINI_MODEL,
        "cases": len(cases),
        "behavior_accuracy": round(sum(c["behavior_match"] for c in cases) / max(1, len(cases)), 3),
        "avg_fact_coverage": avg("fact_coverage"),
        "avg_source_recall": avg("source_recall"),
        "forbidden_claim_cases": sum(1 for c in cases if c["forbidden_violations"]),
        "errors": sum(1 for c in cases if c["status"] == "error"),
        "latency_ms": {"p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95)},
    }


def run(split: str | None, limit: int | None, delay: float) -> dict:
    questions = load_questions(split, limit)
    cases = []
    for q in questions:
        print(f"[{q['id']}] {q['question']}")
        cases.append(evaluate_case(q, query_pipeline(q["question"])))
        time.sleep(delay)
    return {"summary": summarize(cases), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the File Search assistant.")
    parser.add_argument("--live", action="store_true", help="Required: makes real Gemini API calls.")
    parser.add_argument("--split", choices=["development", "holdout"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    args = parser.parse_args()

    if not args.live:
        print("Refusing to run without --live (this evaluator makes real Gemini API calls).")
        sys.exit(2)

    report = run(args.split, args.limit, args.delay)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
