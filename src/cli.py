"""Command-line entry point: `uv run python -m src <command>`.

Commands follow the project milestones:
  inspect-data  -> Milestone 1: validate data, show sample chunks
  retrieve      -> Milestone 1: show retrieval scores for any question
  train         -> Milestone 2: real LoRA training run
  ask           -> Milestone 3: one answer from one system
  evaluate      -> Milestone 4: full benchmark + report tables
"""

import json
import os

import click

from .config import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES, load_settings
from .data import load_documents, load_evaluation_cases, load_training_examples, validate_data
from .generation import SYSTEMS, AnswerPipeline, LocalModelGenerator
from .retrieval import Retriever, chunk_documents

# Windows without Developer Mode cannot create symlinks; the HF cache still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@click.group()
def main() -> None:
    """Northstar hybrid RAG + LoRA support assistant."""


@main.command("inspect-data")
def inspect_data() -> None:
    """Validate policies, training examples and evaluation cases."""

    settings = load_settings()
    documents = load_documents(settings.knowledge_base_dir)
    examples = load_training_examples(settings.training_file)
    cases = load_evaluation_cases(settings.evaluation_file)
    problems = validate_data(documents, examples, cases)

    click.echo(f"Policy sources:    {len(documents)} -> {[d.document_id for d in documents]}")
    click.echo(f"Training examples: {len(examples)}")
    click.echo(f"Evaluation cases:  {len(cases)} "
               f"({sum(not c.should_abstain for c in cases)} answerable, "
               f"{sum(c.should_abstain for c in cases)} should abstain)")
    chunks = chunk_documents(documents)
    click.echo(f"\nChunks: {len(chunks)}. First three, with their metadata:")
    for chunk in chunks[:3]:
        click.echo(f"  {chunk.chunk_id:20} doc={chunk.document_id} pos={chunk.position} "
                   f"title='{chunk.title}'\n    {chunk.text[:90]}...")
    if problems:
        raise click.ClickException("Data problems found:\n  " + "\n  ".join(problems))
    click.echo("\nAll data contracts passed.")


@main.command()
@click.argument("question")
def retrieve(question: str) -> None:
    """Show the top-k chunks and scores for QUESTION (no model involved)."""

    settings = load_settings()
    hits, evidence = Retriever(settings).evidence_for(question)
    for rank, hit in enumerate(hits, 1):
        marker = "PASS" if hit in evidence else "below threshold"
        click.echo(f"{rank}. score={hit.score:.3f} [{hit.document_id}] ({marker}) {hit.text[:80]}...")
    click.echo(f"\nThreshold MIN_SCORE={settings.min_score}: "
               + ("evidence accepted" if evidence else "ABSTAIN (no evidence)"))


@main.command()
def train() -> None:
    """Train the LoRA adapter and save it under artifacts/."""

    from .training import train_adapter

    settings = load_settings()
    click.echo(f"Training LoRA on {settings.base_model} for {settings.train_epochs} epochs...")
    summary = train_adapter(settings, log=click.echo)
    click.echo(f"\nSaved adapter to {summary.adapter_path} in {summary.training_seconds}s "
               f"on {summary.device}. Loss {summary.epoch_losses[0]} -> {summary.epoch_losses[-1]}")


@main.command()
@click.option("--system", type=click.Choice(SYSTEMS), default="rag-lora", show_default=True)
@click.argument("question")
def ask(system: str, question: str) -> None:
    """Answer QUESTION with one system."""

    settings = load_settings()
    retriever = None if system == "base-only" else Retriever(settings)
    generator = LocalModelGenerator(settings, use_adapter=(system == "rag-lora"))
    result = AnswerPipeline(retriever, {system: generator}).answer(question, system)
    click.echo(f"System:   {result.system}")
    click.echo(f"Status:   {result.status}" + (f" ({result.abstain_reason})" if result.abstain_reason else ""))
    click.echo(f"Evidence: {result.evidence_sources} (top score {result.top_score})")
    click.echo(f"Answer:   {result.answer}")
    if result.raw_answer and result.raw_answer != result.answer:
        click.echo(f"Raw model output: {result.raw_answer}")


@main.command()
@click.option("--systems", default=",".join(SYSTEMS), show_default=True,
              help="Comma-separated systems to benchmark.")
def evaluate(systems: str) -> None:
    """Benchmark the systems and write reports/benchmark_results.md."""

    from .evaluation import run_benchmark, write_report

    settings = load_settings()
    chosen = [name.strip() for name in systems.split(",")]
    cases = load_evaluation_cases(settings.evaluation_file)
    retriever = Retriever(settings)
    # The baseline and base-only share one unchanged model; the hybrid loads the adapter.
    generators = {}
    if {"base-only", "rag-base"} & set(chosen):
        base = LocalModelGenerator(settings, use_adapter=False)
        generators.update({"base-only": base, "rag-base": base})
    if "rag-lora" in chosen:
        generators["rag-lora"] = LocalModelGenerator(settings, use_adapter=True)

    results = run_benchmark(AnswerPipeline(retriever, generators), cases, chosen, log=click.echo)
    config = {
        "base_model": f"{settings.base_model} @ {settings.base_model_revision}",
        "embedding_model": f"{settings.embedding_model} @ {settings.embedding_revision}",
        "vector_index": "Qdrant (in-memory, cosine distance), one chunk per paragraph",
        "top_k": settings.top_k,
        "min_score (evidence threshold)": settings.min_score,
        "decoding": f"greedy, max_new_tokens={settings.max_new_tokens}, "
                    f"max_sequence_length={settings.max_sequence_length}",
        "lora": f"targets={LORA_TARGET_MODULES}, r={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}",
    }
    adapter_summary = settings.adapter_path / "training_summary.json"
    if "rag-lora" in chosen and adapter_summary.exists():
        training = json.loads(adapter_summary.read_text(encoding="utf-8"))
        config["training"] = (
            f"epochs={training['epochs']}, lr={training['learning_rate']}, seed={training['random_seed']}, "
            f"trainable={training['trainable_parameters']:,}/{training['total_parameters']:,} "
            f"({training['trainable_percent']}%), loss {training['epoch_losses'][0]} -> "
            f"{training['epoch_losses'][-1]}, {training['training_seconds']}s on {training['device']}"
        )
    output = settings.reports_dir / "benchmark_results.md"
    write_report(results, config, output)
    click.echo(f"\nWrote {output} and {output.with_name('benchmark_trace.json')}")
