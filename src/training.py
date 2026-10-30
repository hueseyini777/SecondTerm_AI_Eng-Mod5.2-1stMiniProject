"""Real local LoRA fine-tuning with Hugging Face PEFT.

Idea in one paragraph: the base model has ~135 million weights. Instead of
changing all of them, LoRA freezes them and adds small extra matrices
("adapters") next to the attention layers. Only those ~0.5 million new
numbers are trained. The result is a small adapter folder that can be loaded
on top of the unchanged base model, or left out to get the baseline back.

Training loop for each example:
  1. feed "prompt + correct answer" into the model,
  2. the model predicts each next token; the loss measures how wrong it was,
     counted ONLY on the answer tokens (prompt tokens are masked with -100),
  3. backpropagation computes how to nudge the adapter weights,
  4. the optimizer (AdamW) applies that nudge.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES, Settings
from .data import load_training_examples
from .prompts import prompt_text, training_text

IGNORE_INDEX = -100  # PyTorch's loss function skips positions with this label


@dataclass
class TrainingSummary:
    base_model: str
    base_model_revision: str | None
    training_data_sha256: str
    prompt_format: str
    lora_target_modules: list[str]
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    max_sequence_length: int
    epochs: int
    learning_rate: float
    batch_size: int
    random_seed: int
    trainable_parameters: int
    total_parameters: int
    trainable_percent: float
    epoch_losses: list[float]
    training_seconds: float
    device: str
    adapter_path: str


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple silicon
        return torch.device("mps")
    return torch.device("cpu")


def encode_example(tokenizer, example, max_length: int):
    """Tokenize one example and hide the prompt part from the loss."""

    full = tokenizer(
        training_text(tokenizer, example.question, example.context, example.answer),
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_ids = tokenizer(
        prompt_text(tokenizer, example.question, example.context),
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    input_ids = full["input_ids"]
    if input_ids.shape[1] > max_length:
        # We refuse to silently cut off the answer - that would teach the wrong thing.
        raise ValueError(f"{example.id} is longer than MAX_SEQUENCE_LENGTH={max_length}")
    if not torch.equal(input_ids[:, : prompt_ids.shape[1]], prompt_ids):
        raise ValueError("Training text must start with the exact inference prompt")

    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = IGNORE_INDEX  # learn the answer, not the prompt
    return input_ids, full["attention_mask"], labels


def train_adapter(settings: Settings, log=print) -> TrainingSummary:
    set_seed(settings.random_seed)  # same seed -> same initial adapter -> repeatable run
    training_bytes = settings.training_file.read_bytes()
    examples = load_training_examples(settings.training_file)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.base_model, revision=settings.base_model_revision
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(tokenizer, ex, settings.max_sequence_length) for ex in examples]

    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model, revision=settings.base_model_revision, dtype=torch.float32
    )
    model.config.use_cache = False  # caching only helps generation, not training

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,  # attention "query" and "value" projections
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    # get_peft_model freezes every original weight and inserts trainable LoRA matrices.
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")

    device = pick_device()
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate
    )

    epoch_losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(1, settings.train_epochs + 1):
        total_loss = 0.0
        for input_ids, attention_mask, labels in encoded:  # batch size 1
            loss = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            ).loss
            loss.backward()  # compute gradients (only adapter weights get them)
            optimizer.step()  # update the adapter weights
            optimizer.zero_grad()
            total_loss += loss.item()
        epoch_losses.append(round(total_loss / len(encoded), 4))
        log(f"Epoch {epoch}/{settings.train_epochs}: mean loss {epoch_losses[-1]:.4f}")
    seconds = time.perf_counter() - started

    # Saves ONLY the small adapter (a few MB), not a copy of the base model.
    settings.adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(settings.adapter_path))
    summary = TrainingSummary(
        base_model=settings.base_model,
        base_model_revision=settings.base_model_revision,
        training_data_sha256=hashlib.sha256(training_bytes).hexdigest(),
        prompt_format="tokenizer chat template: system instruction, user "
        "'Approved context: [source-id] text ... Question: ...', assistant answer",
        lora_target_modules=LORA_TARGET_MODULES,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        max_sequence_length=settings.max_sequence_length,
        epochs=settings.train_epochs,
        learning_rate=settings.learning_rate,
        batch_size=1,
        random_seed=settings.random_seed,
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_percent=round(100 * trainable / total, 3),
        epoch_losses=epoch_losses,
        training_seconds=round(seconds, 1),
        device=str(device),
        adapter_path=str(settings.adapter_path),
    )
    (settings.adapter_path / "training_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary


def load_training_summary(settings: Settings) -> dict:
    """Read the saved summary and refuse adapters trained on different data."""

    path = settings.adapter_path / "training_summary.json"
    if not path.exists():
        raise FileNotFoundError("No trained adapter found. Run: uv run python -m src train")
    summary = json.loads(path.read_text(encoding="utf-8"))
    current = hashlib.sha256(settings.training_file.read_bytes()).hexdigest()
    if summary["training_data_sha256"] != current:
        raise ValueError("Training data changed since the adapter was trained. Retrain it.")
    if summary["base_model"] != settings.base_model:
        raise ValueError("The adapter was trained for a different base model. Retrain it.")
    return summary
