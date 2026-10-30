"""Shared test fixtures.

Tests use a word-hashing embedder and scripted "models" instead of real
downloads, so they run in seconds. They check that the plumbing and safety
rules work; they say nothing about whether LoRA improves answers.
"""

from dataclasses import replace

import pytest

from src.config import load_settings
from src.generation import AnswerPipeline
from src.retrieval import HashEmbedder, Retriever


class ScriptedGenerator:
    """Fake model that always returns the same text and records its calls."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, list]] = []

    def generate(self, question, hits):
        self.calls.append((question, hits))
        return self.text


@pytest.fixture
def settings():
    # The hashing embedder produces lower scores than the real model.
    return replace(load_settings(), min_score=0.2)


@pytest.fixture
def retriever(settings):
    return Retriever(settings, embedder=HashEmbedder())


@pytest.fixture
def make_pipeline(retriever):
    def build(text: str, system: str = "rag-lora"):
        generator = ScriptedGenerator(text)
        return AnswerPipeline(retriever, {system: generator}), generator

    return build
