"""Evaluation suite for the assistant.

Two layers:

1. **Retrieval evals** (free, deterministic): for each golden question, does the
   top-k retrieval include a chunk from the expected source document?
   Reports hit-rate@k and MRR. Runs in CI on every push.

2. **Answer evals** (cost real tokens): generate an answer, check expected facts
   appear, then ask a judge model to score faithfulness (is every claim
   supported by the retrieved passages?) and relevance, using structured
   outputs so scores are machine-readable. Skipped unless ANTHROPIC_API_KEY is
   set, or when ``--retrieval-only`` is passed.

Exit code is non-zero if any gate fails, so the script can block a merge.

Usage:
    python -m evals.run_evals                 # both layers (needs API key)
    python -m evals.run_evals --retrieval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import settings
from app.prompts import PROMPT_VERSION, format_context
from app.retriever import build_retriever

GOLDEN = Path(__file__).parent / "golden_set.jsonl"
REPORT = Path(__file__).parent / "last_report.json"

RETRIEVAL_HIT_GATE = 0.85
FAITHFULNESS_GATE = 4.0
RELEVANCE_GATE = 4.0


class JudgeVerdict(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="5 = every claim is supported by the passages; 1 = mostly unsupported")
    relevance: int = Field(ge=1, le=5, description="5 = fully answers the question; 1 = off topic")
    unsupported_claims: list[str] = Field(description="Claims in the answer not supported by the passages")
    rationale: str


JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented assistant. You receive the passages the
assistant saw, the visitor's question and the assistant's answer. Score faithfulness (are all factual
claims in the answer supported by the passages?) and relevance (does the answer address the
question?). Citations like [1] are expected and are not claims. Saying "I don't have that detail" is
faithful. List every unsupported claim verbatim."""


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_eval(golden: list[dict]) -> dict:
    retriever = build_retriever()
    hits, rr = 0, 0.0
    rows = []
    for item in golden:
        results = retriever.search(item["question"], settings.top_k)
        sources = [c.source_id for c, _ in results]
        rank = next((i + 1 for i, s in enumerate(sources) if s in item["expected_sources"]), None)
        if rank:
            hits += 1
            rr += 1 / rank
        rows.append({"id": item["id"], "hit": bool(rank), "rank": rank, "retrieved": sources})
    n = len(golden)
    return {"hit_rate": hits / n, "mrr": rr / n, "k": settings.top_k, "rows": rows}


def answer_eval(golden: list[dict]) -> dict:
    import anthropic

    from app.engine import RagEngine

    client = anthropic.Anthropic()
    engine = RagEngine(client=client)
    rows, total_cost = [], 0.0
    for item in golden:
        answer, _, trace = engine.answer(item["question"])
        total_cost += trace.get("cost_usd", 0.0)
        chunks = [c for c, _ in engine.retriever.search(item["question"], settings.top_k)]
        facts_ok = all(f.lower() in answer.lower() for f in item["expected_facts"])

        judged = client.messages.parse(
            model=settings.judge_model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            output_format=JudgeVerdict,
            messages=[
                {
                    "role": "user",
                    "content": f"{format_context(chunks)}\n\nQuestion: {item['question']}\n\nAnswer:\n{answer}",
                }
            ],
        )
        verdict = judged.parsed_output
        rows.append(
            {
                "id": item["id"],
                "facts_ok": facts_ok,
                "faithfulness": verdict.faithfulness,
                "relevance": verdict.relevance,
                "unsupported": verdict.unsupported_claims,
                "answer": answer,
            }
        )
        print(f"  {item['id']}: facts={'ok' if facts_ok else 'MISS'} faith={verdict.faithfulness} rel={verdict.relevance}")
    n = len(rows)
    return {
        "facts_pass_rate": sum(r["facts_ok"] for r in rows) / n,
        "avg_faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "avg_relevance": sum(r["relevance"] for r in rows) / n,
        "generation_cost_usd": round(total_cost, 4),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    golden = load_golden()
    report: dict = {"prompt_version": PROMPT_VERSION, "model": settings.model, "n": len(golden)}
    failed = False

    print(f"Retrieval eval on {len(golden)} questions (backend={settings.retriever_backend}, k={settings.top_k})")
    r = retrieval_eval(golden)
    report["retrieval"] = r
    print(f"  hit-rate@{r['k']} = {r['hit_rate']:.2f}   MRR = {r['mrr']:.2f}")
    for row in r["rows"]:
        if not row["hit"]:
            print(f"  MISS {row['id']}: got {row['retrieved']}")
    if r["hit_rate"] < RETRIEVAL_HIT_GATE:
        print(f"  FAIL: hit-rate below gate {RETRIEVAL_HIT_GATE}")
        failed = True

    if not args.retrieval_only:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Answer eval skipped: ANTHROPIC_API_KEY not set")
        else:
            print(f"Answer eval with judge {settings.judge_model}")
            a = answer_eval(golden)
            report["answers"] = a
            print(
                f"  facts pass = {a['facts_pass_rate']:.2f}  faithfulness = {a['avg_faithfulness']:.2f}  "
                f"relevance = {a['avg_relevance']:.2f}  cost = ${a['generation_cost_usd']}"
            )
            if a["avg_faithfulness"] < FAITHFULNESS_GATE or a["avg_relevance"] < RELEVANCE_GATE:
                print("  FAIL: judge scores below gate")
                failed = True

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {REPORT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
