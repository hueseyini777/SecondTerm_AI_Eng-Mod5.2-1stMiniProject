"""Answer paths: abstention, citation validation, and shared retrieval."""

import pytest

from src.generation import AnswerPipeline
from src.prompts import FALLBACK_ANSWER
from tests.conftest import ScriptedGenerator

SHIPPING_QUESTION = "What is the usual standard shipping time?"


def test_valid_cited_answer_is_served(make_pipeline):
    pipeline, _ = make_pipeline("Standard shipping takes three to five business days. [shipping]")
    result = pipeline.answer(SHIPPING_QUESTION, "rag-lora")
    assert result.status == "answered"
    assert result.citations == ["shipping"]


def test_unsupported_question_abstains_without_calling_the_model(make_pipeline):
    pipeline, generator = make_pipeline("Invented answer. [shipping]")
    result = pipeline.answer("Where can I find next week's cafeteria menu?", "rag-lora")
    assert result.status == "abstained"
    assert result.abstain_reason == "below-threshold"
    assert result.answer == FALLBACK_ANSWER
    assert generator.calls == []


def test_model_can_decline_even_with_evidence(make_pipeline):
    pipeline, _ = make_pipeline(FALLBACK_ANSWER)
    result = pipeline.answer(SHIPPING_QUESTION, "rag-lora")
    assert result.status == "abstained"
    assert result.abstain_reason == "model-declined"


@pytest.mark.parametrize(
    "text",
    [
        "Standard shipping takes three to five business days.",  # no citation
        "Support can make an exception. [executive-override]",  # invented source
        "Three to five days. [shipping] [refunds]",  # refunds was not retrieved
        "[shipping]",  # citation with no content
    ],
)
def test_invalid_citations_are_rejected_and_not_served(make_pipeline, text):
    pipeline, _ = make_pipeline(text)
    result = pipeline.answer(SHIPPING_QUESTION, "rag-lora")
    assert result.status == "rejected"
    assert result.raw_answer == text
    assert text not in result.answer


def test_baseline_and_hybrid_receive_identical_evidence(retriever):
    base, lora = ScriptedGenerator("x [shipping]"), ScriptedGenerator("y [shipping]")
    pipeline = AnswerPipeline(retriever, {"rag-base": base, "rag-lora": lora})
    pipeline.answer(SHIPPING_QUESTION, "rag-base")
    pipeline.answer(SHIPPING_QUESTION, "rag-lora")
    assert base.calls == lora.calls


def test_base_only_gets_no_evidence(retriever):
    generator = ScriptedGenerator("Probably a week.")
    result = AnswerPipeline(retriever, {"base-only": generator}).answer(SHIPPING_QUESTION, "base-only")
    assert generator.calls == [(SHIPPING_QUESTION, [])]
    assert result.status == "answered" and result.citations == []
