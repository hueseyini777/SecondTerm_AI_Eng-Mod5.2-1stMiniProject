"""Data contracts: the three data layers must be valid and kept separate."""

import json

import pytest

from src.data import (
    EvaluationCase,
    TrainingExample,
    citations_in,
    load_documents,
    load_evaluation_cases,
    load_training_examples,
    validate_data,
)


@pytest.fixture
def real_data(settings):
    return (
        load_documents(settings.knowledge_base_dir),
        load_training_examples(settings.training_file),
        load_evaluation_cases(settings.evaluation_file),
    )


def test_supplied_data_passes_all_contracts(real_data):
    documents, examples, cases = real_data
    assert validate_data(documents, examples, cases) == []
    assert {d.document_id for d in documents} >= {"refunds", "shipping", "account-security"}
    assert any(case.should_abstain for case in cases)


def test_duplicate_evaluation_ids_are_reported(real_data):
    documents, examples, cases = real_data
    problems = validate_data(documents, examples, cases + [cases[0]])
    assert any("Duplicate evaluation IDs" in p for p in problems)


def test_unknown_source_in_training_is_reported(real_data):
    documents, examples, cases = real_data
    bad = TrainingExample("bad-1", ["no-such-policy"], "ctx", "q?", "a [no-such-policy]")
    problems = validate_data(documents, examples + [bad], cases)
    assert any("unknown sources" in p for p in problems)


def test_evaluation_question_leaking_into_training_is_reported(real_data):
    documents, examples, cases = real_data
    leaked = TrainingExample("leak-1", ["shipping"], "[shipping] ...", cases[0].question, "a [shipping]")
    problems = validate_data(documents, examples + [leaked], cases)
    assert any("leaked" in p for p in problems)


def test_unanswerable_case_may_not_expect_sources(real_data):
    documents, examples, cases = real_data
    broken = EvaluationCase("x", "q?", ["refunds"], [], [], should_abstain=True)
    assert any("should abstain" in p for p in validate_data(documents, examples, cases + [broken]))


def test_training_record_missing_a_field_fails(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"id": "a", "context": "c", "question": "q", "answer": "x"}))
    with pytest.raises(ValueError, match="missing"):
        load_training_examples(path)


def test_citations_are_extracted_in_order_without_duplicates():
    assert citations_in("A [refunds]. B [shipping] [refunds]") == ["refunds", "shipping"]
