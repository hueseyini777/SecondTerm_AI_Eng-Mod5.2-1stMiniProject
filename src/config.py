"""All tunable settings in one place, read from `.env` with safe defaults.

Keeping every "knob" here (model names, top-k, threshold, LoRA training
settings) means the evaluation report can print the exact configuration that
produced its numbers.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pinning a model "revision" (a git commit on the Hugging Face Hub) means
# everyone downloads byte-identical weights, so results are reproducible.
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# LoRA hyper-parameters (see docs/LEARNING_GUIDE.md for what each one means).
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


@dataclass(frozen=True)
class Settings:
    knowledge_base_dir: Path
    training_file: Path
    evaluation_file: Path
    adapter_path: Path
    reports_dir: Path
    base_model: str
    base_model_revision: str | None
    embedding_model: str
    embedding_revision: str | None
    top_k: int
    min_score: float
    max_sequence_length: int
    max_new_tokens: int
    train_epochs: int
    learning_rate: float
    random_seed: int

    def __post_init__(self) -> None:
        if self.top_k <= 0 or self.max_new_tokens <= 0 or self.train_epochs <= 0:
            raise ValueError("TOP_K, MAX_NEW_TOKENS and TRAIN_EPOCHS must be positive")
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError("MIN_SCORE is a cosine similarity and must be in [-1, 1]")
        if self.learning_rate <= 0:
            raise ValueError("LEARNING_RATE must be positive")


def load_settings(root: Path = PROJECT_ROOT) -> Settings:
    """Read `.env` (if present) and fall back to the documented defaults."""

    load_dotenv(root / ".env")
    base_model = os.getenv("BASE_MODEL") or DEFAULT_BASE_MODEL
    embedding_model = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    return Settings(
        knowledge_base_dir=root / "data" / "knowledge_base",
        training_file=root / "data" / "training" / "instruction_examples.jsonl",
        evaluation_file=root / "data" / "evaluation" / "cases.json",
        adapter_path=root / (os.getenv("ADAPTER_PATH") or "artifacts/support-lora-adapter"),
        reports_dir=root / "reports",
        base_model=base_model,
        # Only use the pinned revision when the default model is selected.
        base_model_revision=os.getenv("BASE_MODEL_REVISION")
        or (DEFAULT_BASE_REVISION if base_model == DEFAULT_BASE_MODEL else None),
        embedding_model=embedding_model,
        embedding_revision=os.getenv("EMBEDDING_REVISION")
        or (DEFAULT_EMBEDDING_REVISION if embedding_model == DEFAULT_EMBEDDING_MODEL else None),
        top_k=int(os.getenv("TOP_K") or 3),
        min_score=float(os.getenv("MIN_SCORE") or 0.30),
        max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH") or 256),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS") or 80),
        train_epochs=int(os.getenv("TRAIN_EPOCHS") or 8),
        learning_rate=float(os.getenv("LEARNING_RATE") or 0.0002),
        random_seed=int(os.getenv("RANDOM_SEED") or 42),
    )
