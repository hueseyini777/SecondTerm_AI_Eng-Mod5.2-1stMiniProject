"""Metric rubrics and report generation."""

import json

from src.data import EvaluationCase
from src.evaluation import contains_term, run_benchmark, score_answer, style_checks, write_report
from src.generation import Answer, AnswerPipeline
from tests.conftest import ScriptedGenerator

SUPPORTED = EvaluationCase("s", "q?", ["shipping"], ["three to five", "cannot"], ["can change after packing"], False)
UNSUPPORTED = EvaluationCase("u", "menu?", [], [], [], True)


def test_terms_match_whole_phrases_only():
    assert contains_term("Support cannot do that.", "cannot")
    assert not contains_term("Read the note.", "not")


def test_supported_answer_scoring():
    answer = Answer("rag-lora", "q?", "answered", "It takes three to five days. [shipping]",
                    raw_answer="It takes three to five days. [shipping]", citations=["shipping"])
    row = score_answer(SUPPORTED, answer)
    assert row["relevance"] == 0.5
    assert row["faithful"] is True
    assert row["decision_correct"] is True


def test_uncited_or_forbidden_answers_are_not_faithful():
    uncited = Answer("base-only", "q?", "answered", "Three to five days.", raw_answer="x")
    forbidden = Answer("rag-lora", "q?", "answered", "Support can change after packing. [shipping]",
                       raw_answer="x", citations=["shipping"])
    assert score_answer(SUPPORTED, uncited)["faithful"] is False
    assert score_answer(SUPPORTED, forbidden)["faithful"] is False


def test_answering_an_unsupported_question_is_a_wrong_decision():
    answered = Answer("rag-lora", "menu?", "answered", "Upstairs. [shipping]", citations=["shipping"])
    abstained = Answer("rag-lora", "menu?", "abstained", "insufficient")
    assert score_answer(UNSUPPORTED, answered)["decision_correct"] is False
    assert score_answer(UNSUPPORTED, answered)["faithful"] is False
    assert score_answer(UNSUPPORTED, abstained)["decision_correct"] is True


def test_rejected_answer_scores_zero_relevance_but_keeps_raw_coverage():
    rejected = Answer("rag-base", "q?", "rejected", "failed validation",
                      raw_answer="Three to five days; support cannot change it.")
    row = score_answer(SUPPORTED, rejected)
    assert row["relevance"] == 0.0
    assert row["raw_coverage"] == 1.0


def test_style_checks_flag_hedging_and_repetition():
    assert all(style_checks("Collect the invoice number. Then escalate. [refunds]").values())
    checks = style_checks("Maybe wait. Maybe wait.")
    assert not checks["no_hedging"] and not checks["no_repetition"]


def test_report_contains_required_sections(retriever, settings, tmp_path):
    pipeline = AnswerPipeline(retriever, {
        "rag-base": ScriptedGenerator("No citation here."),
        "rag-lora": ScriptedGenerator("Standard shipping takes three to five business days. [shipping]"),
    })
    cases = [
        EvaluationCase("ship", "What is the usual standard shipping time?", ["shipping"], ["three to five"], [], False),
        EvaluationCase("menu", "Where can I find next week's cafeteria menu?", [], [], [], True),
    ]
    results = run_benchmark(pipeline, cases, ["rag-base", "rag-lora"], log=lambda _: None)
    output = tmp_path / "benchmark_results.md"
    write_report(results, {"top_k": settings.top_k}, output)

    text = output.read_text(encoding="utf-8")
    for heading in ("## Retrieval", "Threshold sweep", "## Answer quality per system",
                    "Hybrid minus baseline", "## Per-case comparison", "rejected answers"):
        assert heading in text
    trace = json.loads(output.with_name("benchmark_trace.json").read_text(encoding="utf-8"))
    assert trace["systems"]["rag-lora"]["summary"]["answered_supported"] == "1/1"
    assert trace["systems"]["rag-base"]["summary"]["rejected"] == 1
