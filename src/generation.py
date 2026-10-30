"""The "G" in RAG, plus the safety checks shared by every system.

Three systems are compared:
  - base-only : base model, no retrieval        (bonus: shows what retrieval adds)
  - rag-base  : retrieval + unchanged base model (the BASELINE)
  - rag-lora  : retrieval + base model + LoRA    (the HYBRID)

rag-base and rag-lora get identical evidence, prompt, threshold and
validation. The only difference is whether the adapter is loaded.

Every result ends in exactly one status:
  answered  - the answer cites only sources that were actually retrieved
  abstained - no evidence above the threshold, or the model said the
              approved material is insufficient
  rejected  - the model answered, but its citations failed validation, so
              the text is NOT shown to the customer
"""

import time
from dataclasses import dataclass, field
from typing import Protocol

import torch

from .config import Settings
from .data import CITATION_PATTERN, citations_in
from .prompts import FALLBACK_ANSWER, format_evidence, prompt_text
from .retrieval import Retriever, SearchHit

SYSTEMS = ("base-only", "rag-base", "rag-lora")
REJECTION_MESSAGE = "The generated answer failed citation validation. Escalate for manual review."


@dataclass
class Answer:
    system: str
    question: str
    status: str  # "answered" | "abstained" | "rejected"
    answer: str  # what would be shown to the customer
    raw_answer: str | None = None  # what the model actually wrote (for analysis)
    citations: list[str] = field(default_factory=list)
    evidence_sources: list[str] = field(default_factory=list)
    top_score: float | None = None
    abstain_reason: str | None = None  # "below-threshold" | "model-declined"
    latency_seconds: float = 0.0


class Generator(Protocol):
    def generate(self, question: str, hits: list[SearchHit]) -> str: ...


class LocalModelGenerator:
    """Runs SmolLM2 on this computer, with or without the LoRA adapter."""

    def __init__(self, settings: Settings, use_adapter: bool) -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.base_model, revision=settings.base_model_revision
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            settings.base_model, revision=settings.base_model_revision
        )
        if use_adapter:
            from .training import load_training_summary

            load_training_summary(settings)  # refuses stale or missing adapters
            # Wrap the frozen base model with the small trained adapter.
            model = PeftModel.from_pretrained(model, str(settings.adapter_path))
        self.model = model.eval()

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        prompt = prompt_text(self.tokenizer, question, format_evidence(hits))
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if encoded["input_ids"].shape[1] > self.settings.max_sequence_length:
            raise ValueError("Prompt is longer than MAX_SEQUENCE_LENGTH; refusing to truncate it")
        with torch.no_grad():  # no training here, so no gradients needed
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,  # greedy decoding = the same answer every run
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def is_abstention(text: str) -> bool:
    return "approved material is insufficient" in text.lower()


def citations_are_valid(text: str, allowed_sources: set[str]) -> bool:
    """Accept only answers with real content and >=1 citation, all retrieved."""

    citations = citations_in(text)
    has_content = bool(CITATION_PATTERN.sub("", text).strip())
    return has_content and bool(citations) and set(citations) <= allowed_sources


class AnswerPipeline:
    """Routes a question through one of the three systems."""

    def __init__(
        self,
        retriever: Retriever | None,
        generators: dict[str, Generator],
    ) -> None:
        self.retriever = retriever
        self.generators = generators  # e.g. {"rag-base": ..., "rag-lora": ...}

    def answer(self, question: str, system: str) -> Answer:
        started = time.perf_counter()
        result = self._answer(question, system)
        result.latency_seconds = time.perf_counter() - started
        return result

    def _answer(self, question: str, system: str) -> Answer:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system {system!r}; choose from {SYSTEMS}")
        generator = self.generators[system]

        if system == "base-only":
            text = generator.generate(question, [])
            status = "abstained" if is_abstention(text) else "answered"
            return Answer(system, question, status, text, raw_answer=text,
                          citations=citations_in(text),
                          abstain_reason="model-declined" if status == "abstained" else None)

        # Step 1: shared retrieval + shared evidence threshold.
        hits, evidence = self.retriever.evidence_for(question)
        top_score = hits[0].score if hits else None
        sources = list(dict.fromkeys(hit.document_id for hit in evidence))
        if not evidence:
            return Answer(system, question, "abstained", FALLBACK_ANSWER,
                          top_score=top_score, abstain_reason="below-threshold")

        # Step 2: generate from the evidence.
        text = generator.generate(question, evidence)
        common = dict(raw_answer=text, evidence_sources=sources, top_score=top_score)
        if is_abstention(text):
            return Answer(system, question, "abstained", FALLBACK_ANSWER,
                          abstain_reason="model-declined", **common)

        # Step 3: never trust the model's citations without checking them.
        if not citations_are_valid(text, set(sources)):
            return Answer(system, question, "rejected", REJECTION_MESSAGE, **common)
        return Answer(system, question, "answered", text, citations=citations_in(text), **common)
