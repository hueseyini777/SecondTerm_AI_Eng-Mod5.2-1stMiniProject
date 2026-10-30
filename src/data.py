"""Load and validate the three separate data layers.

1. Policy sources   (data/knowledge_base/*.md)  -> what the assistant may cite
2. Training examples (data/training/*.jsonl)     -> what the LoRA adapter learns from
3. Evaluation cases  (data/evaluation/cases.json) -> the held-out exam

We validate *before* training so a typo (e.g. citing a policy that does not
exist) fails loudly instead of silently producing a worse model.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

# A citation is anything written in square brackets, e.g. "[refunds]".
CITATION_PATTERN = re.compile(r"\[([^\[\]]*)\]")


@dataclass
class PolicyDocument:
    document_id: str  # the filename without ".md", e.g. "refunds"
    title: str  # the first "# Heading" line
    text: str  # everything after the heading


@dataclass
class TrainingExample:
    id: str
    reference_ids: list[str]
    context: str
    question: str
    answer: str


@dataclass
class EvaluationCase:
    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    forbidden_terms: list[str]
    should_abstain: bool


def citations_in(text: str) -> list[str]:
    """Return the unique source IDs cited in a piece of text, in order."""

    return list(dict.fromkeys(match.strip() for match in CITATION_PATTERN.findall(text)))


def load_documents(folder: Path) -> list[PolicyDocument]:
    documents = []
    for path in sorted(folder.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path.name} must start with a '# Title' line")
        documents.append(
            PolicyDocument(
                document_id=path.stem,
                title=lines[0].removeprefix("# ").strip(),
                text="\n".join(lines[1:]).strip(),
            )
        )
    if not documents:
        raise ValueError(f"No policy documents found in {folder}")
    return documents


def load_training_examples(path: Path) -> list[TrainingExample]:
    required = {"id", "reference_ids", "context", "question", "answer"}
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = required - record.keys()
        empty = [key for key in required & record.keys() if not record[key]]
        if missing or empty:
            raise ValueError(
                f"Training line {line_number} is missing {sorted(missing)} "
                f"or has empty {sorted(empty)}"
            )
        examples.append(TrainingExample(**{key: record[key] for key in required}))
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    cases = [EvaluationCase(**record) for record in json.loads(path.read_text(encoding="utf-8"))]
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def validate_data(
    documents: list[PolicyDocument],
    examples: list[TrainingExample],
    cases: list[EvaluationCase],
) -> list[str]:
    """Check the data contracts and return a list of human-readable problems.

    An empty list means everything is consistent.
    """

    problems: list[str] = []
    source_ids = {document.document_id for document in documents}

    for label, ids in (
        ("training", [example.id for example in examples]),
        ("evaluation", [case.id for case in cases]),
    ):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            problems.append(f"Duplicate {label} IDs: {duplicates}")

    for example in examples:
        unknown = set(example.reference_ids) - source_ids
        if unknown:
            problems.append(f"{example.id} references unknown sources {sorted(unknown)}")
        # The answer may only cite sources that were shown in its context.
        bad_citations = set(citations_in(example.answer)) - set(example.reference_ids)
        if bad_citations:
            problems.append(f"{example.id} answer cites {sorted(bad_citations)} not in its context")

    for case in cases:
        unknown = set(case.expected_sources) - source_ids
        if unknown:
            problems.append(f"{case.id} expects unknown sources {sorted(unknown)}")
        if case.should_abstain and case.expected_sources:
            problems.append(f"{case.id} should abstain but lists expected sources")

    # Held-out means held out: no evaluation question may also be a training question.
    training_questions = {example.question.strip().casefold() for example in examples}
    leaked = [case.id for case in cases if case.question.strip().casefold() in training_questions]
    if leaked:
        problems.append(f"Evaluation questions leaked into training: {leaked}")

    return problems
