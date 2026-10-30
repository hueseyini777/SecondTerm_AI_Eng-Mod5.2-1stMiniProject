"""Benchmark: run every system on the same held-out cases with the same rubric.

Metrics are kept in separate families so one good number cannot hide a bad one:

RETRIEVAL (measured once, shared by both RAG systems)
  recall@k  - for answerable cases: was the expected policy in the top-k?
  MRR       - mean reciprocal rank: 1.0 if it was ranked 1st, 0.5 if 2nd, ...

ANSWERS (measured per system)
  relevance    - share of required terms present in the served answer
                 (answerable cases; abstained/rejected answers score 0)
  faithfulness - of the answers actually served: cites only expected
                 sources and contains no forbidden claim. Answering an
                 unanswerable question always counts as unfaithful.
  style        - of the answers served: short direct first sentence, short
                 overall, no hedging words, no repeated sentences
  decision     - answered when it should, abstained when it should
  latency      - seconds per answer (model loading excluded)

All rubrics are deterministic word checks: transparent and free, but blind to
synonyms. That is why raw answers are saved and read by hand as well.
"""

import json
import re
from pathlib import Path
from statistics import mean

from .data import EvaluationCase
from .generation import Answer, AnswerPipeline
from .retrieval import Retriever

HEDGE_WORDS = ("maybe", "perhaps", "probably", "possibly", "might")
THRESHOLD_SWEEP = (0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70)


# ---------------------------------------------------------------- retrieval
def evaluate_retrieval(retriever: Retriever, cases: list[EvaluationCase]) -> dict:
    rows = []
    for case in cases:
        hits = retriever.search(case.question)
        ranked = list(dict.fromkeys(hit.document_id for hit in hits))
        rank = next((i for i, src in enumerate(ranked, 1) if src in case.expected_sources), None)
        rows.append({
            "id": case.id,
            "should_abstain": case.should_abstain,
            "expected": case.expected_sources,
            "retrieved": ranked,
            "scores": [round(hit.score, 3) for hit in hits],
            "top_score": round(hits[0].score, 3) if hits else None,
            "first_relevant_rank": rank,
        })
    supported = [row for row in rows if not row["should_abstain"]]
    unsupported = [row for row in rows if row["should_abstain"]]
    return {
        "recall_at_k": mean(float(row["first_relevant_rank"] is not None) for row in supported),
        "mrr": mean(1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0.0
                    for row in supported),
        "min_supported_top_score": min(row["top_score"] for row in supported),
        "max_unsupported_top_score": max(row["top_score"] for row in unsupported),
        "sweep": threshold_sweep(rows),
        "rows": rows,
    }


def threshold_sweep(rows: list[dict]) -> list[dict]:
    """How would a retrieval-only answer/abstain rule do at each threshold?

    A diagnostic, not a tuning step: choosing the threshold from these same
    cases and then reporting their score would leak the exam into the design.
    """

    sweep = []
    for threshold in THRESHOLD_SWEEP:
        sweep.append({
            "threshold": threshold,
            "supported_kept": sum(r["top_score"] >= threshold for r in rows if not r["should_abstain"]),
            "unsupported_blocked": sum(r["top_score"] < threshold for r in rows if r["should_abstain"]),
        })
    return sweep


# ------------------------------------------------------------------ answers
def contains_term(text: str, term: str) -> bool:
    """Whole-phrase, case-insensitive match ('not' does not match 'note')."""

    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def term_coverage(text: str | None, terms: list[str]) -> float:
    if not terms:
        return 1.0
    if not text:
        return 0.0
    return sum(contains_term(text, term) for term in terms) / len(terms)


def style_checks(text: str) -> dict[str, bool]:
    without_citations = re.sub(r"\[[^\]]*\]", "", text).strip()
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", without_citations) if s.strip()]
    first = sentences[0] if sentences else ""
    return {
        "direct_first_sentence": 0 < len(first.split()) <= 25,
        "concise": len(without_citations.split()) <= 60,
        "no_hedging": not any(contains_term(text, word) for word in HEDGE_WORDS),
        "no_repetition": len(sentences) == len(set(sentences)),
    }


def score_answer(case: EvaluationCase, result: Answer) -> dict:
    answered = result.status == "answered"
    if case.should_abstain:
        faithful = None if not answered else False
    else:
        # An empty citation list is a subset of anything, so require >= 1 citation.
        faithful = None if not answered else (
            bool(result.citations)
            and set(result.citations) <= set(case.expected_sources)
            and not any(contains_term(result.answer, t) for t in case.forbidden_terms)
        )
    checks = style_checks(result.answer) if answered else None
    return {
        "id": case.id,
        "system": result.system,
        "should_abstain": case.should_abstain,
        "status": result.status,
        "abstain_reason": result.abstain_reason,
        "decision_correct": (result.status == "abstained") if case.should_abstain else answered,
        "relevance": None if case.should_abstain else (
            term_coverage(result.answer, case.required_terms) if answered else 0.0),
        # Diagnostic: did the raw model text contain the facts, even if rejected?
        "raw_coverage": None if case.should_abstain else term_coverage(result.raw_answer, case.required_terms),
        "faithful": faithful,
        "style": None if checks is None else sum(checks.values()) / len(checks),
        "style_checks": checks,
        "citations": result.citations,
        "answer": result.answer,
        "raw_answer": result.raw_answer,
        "top_score": result.top_score,
        "latency_seconds": round(result.latency_seconds, 3),
    }


def _mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return mean(values) if values else None


def summarize(rows: list[dict]) -> dict:
    supported = [r for r in rows if not r["should_abstain"]]
    unsupported = [r for r in rows if r["should_abstain"]]
    return {
        "relevance": _mean(r["relevance"] for r in supported),
        "raw_coverage": _mean(r["raw_coverage"] for r in supported),
        "faithfulness": _mean(r["faithful"] for r in rows),
        "faithful_count": f"{sum(r['faithful'] is True for r in rows)}/{sum(r['faithful'] is not None for r in rows)}",
        "style": _mean(r["style"] for r in rows),
        "decision_accuracy": _mean(r["decision_correct"] for r in rows),
        "answered_supported": f"{sum(r['status'] == 'answered' for r in supported)}/{len(supported)}",
        "abstained_unsupported": f"{sum(r['status'] == 'abstained' for r in unsupported)}/{len(unsupported)}",
        "rejected": sum(r["status"] == "rejected" for r in rows),
        "mean_latency": _mean(r["latency_seconds"] for r in rows),
    }


def run_benchmark(pipeline: AnswerPipeline, cases: list[EvaluationCase], systems, log=print) -> dict:
    results = {"retrieval": evaluate_retrieval(pipeline.retriever, cases), "systems": {}}
    for system in systems:
        rows = []
        for case in cases:
            rows.append(score_answer(case, pipeline.answer(case.question, system)))
            log(f"  {system:9} {case.id:28} -> {rows[-1]['status']}")
        results["systems"][system] = {"summary": summarize(rows), "rows": rows}
    return results


# ------------------------------------------------------------------ report
def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _short(text: str | None, limit: int = 110) -> str:
    text = (text or "-").replace("\n", " ").replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_report(results: dict, config: dict, path: Path) -> None:
    """Write the auto-generated metrics tables (Markdown) and a full JSON trace."""

    retrieval = results["retrieval"]
    systems = results["systems"]
    lines = ["# Benchmark Results (auto-generated)", "",
             "Generated by `uv run python -m src evaluate`. The hand-written analysis is in "
             "[evaluation_report.md](evaluation_report.md).", "", "## Configuration", ""]
    lines += [f"- **{key}**: `{value}`" for key, value in config.items()]

    lines += ["", "## Retrieval (shared by both RAG systems)", "",
              "| Metric | Value |", "| --- | ---: |",
              f"| Recall@k (supported cases) | {_fmt(retrieval['recall_at_k'])} |",
              f"| Mean reciprocal rank | {_fmt(retrieval['mrr'])} |",
              f"| Weakest top score, supported | {_fmt(retrieval['min_supported_top_score'])} |",
              f"| Strongest top score, unsupported | {_fmt(retrieval['max_unsupported_top_score'])} |",
              "", "| Case | Should abstain | Expected | Retrieved (rank order) | Scores | First relevant rank |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for row in retrieval["rows"]:
        lines.append(f"| {row['id']} | {_fmt(row['should_abstain'])} | {', '.join(row['expected']) or '-'} "
                     f"| {', '.join(row['retrieved'])} | {', '.join(map(str, row['scores']))} "
                     f"| {_fmt(row['first_relevant_rank'])} |")

    lines += ["", "### Threshold sweep (diagnostic only)", "",
              "| Threshold | Supported cases kept | Unsupported cases blocked |", "| ---: | ---: | ---: |"]
    n_sup = sum(not r["should_abstain"] for r in retrieval["rows"])
    n_uns = len(retrieval["rows"]) - n_sup
    for row in retrieval["sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['supported_kept']}/{n_sup} | {row['unsupported_blocked']}/{n_uns} |")

    lines += ["", "## Answer quality per system", "",
              "| System | Relevance | Raw coverage (pre-validation) | Faithfulness (served answers) | Style | "
              "Decision accuracy | Answered supported | Abstained unsupported | Rejected | Mean latency (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in systems.items():
        s = data["summary"]
        lines.append(f"| {name} | {_fmt(s['relevance'])} | {_fmt(s['raw_coverage'])} | {_fmt(s['faithfulness'])} "
                     f"({s['faithful_count']}) | {_fmt(s['style'])} | {_fmt(s['decision_accuracy'])} "
                     f"| {s['answered_supported']} | {s['abstained_unsupported']} | {s['rejected']} "
                     f"| {_fmt(s['mean_latency'])} |")

    if {"rag-base", "rag-lora"} <= systems.keys():
        base, lora = systems["rag-base"]["summary"], systems["rag-lora"]["summary"]
        lines += ["", "### Hybrid minus baseline", "", "| Metric | Delta |", "| --- | ---: |"]
        for key in ("relevance", "raw_coverage", "faithfulness", "style", "decision_accuracy", "mean_latency"):
            if base[key] is not None and lora[key] is not None:
                lines.append(f"| {key} | {lora[key] - base[key]:+.3f} |")

    lines += ["", "## Per-case comparison", "",
              "| Case | System | Status | Relevance | Raw coverage | Faithful | Style | Served answer |",
              "| --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    case_ids = [row["id"] for row in retrieval["rows"]]
    for case_id in case_ids:
        for name, data in systems.items():
            r = next(row for row in data["rows"] if row["id"] == case_id)
            status = r["status"] + (f" ({r['abstain_reason']})" if r["abstain_reason"] else "")
            lines.append(f"| {case_id} | {name} | {status} | {_fmt(r['relevance'])} | {_fmt(r['raw_coverage'])} "
                         f"| {_fmt(r['faithful'])} | {_fmt(r['style'])} | {_short(r['answer'])} |")

    lines += ["", "## Raw model outputs for rejected answers", "",
              "Rejected text is never shown to customers; it is listed only to diagnose *why* it failed.", ""]
    for name, data in systems.items():
        for r in data["rows"]:
            if r["status"] == "rejected":
                lines.append(f"- `{name}` / `{r['id']}`: {_short(r['raw_answer'], 220)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_name("benchmark_trace.json").write_text(
        json.dumps({"config": config, **results}, indent=2, default=str), encoding="utf-8"
    )
