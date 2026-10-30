"""The ONE prompt format shared by training and by both answer paths.

If training used one layout and inference another, the adapter would be
tested on something it never saw. Sharing this module prevents that.

What the model sees (rendered with SmolLM2's chat template):

    <|im_start|>system
    Answer only from approved context. ... <|im_end|>
    <|im_start|>user
    Approved context:
    [shipping] A delivery address can be changed only before ...
    Question: Can support change ...<|im_end|>
    <|im_start|>assistant
    <- the model writes its answer here
"""

FALLBACK_ANSWER = "The approved material is insufficient. Escalate for manual review."

SYSTEM_INSTRUCTION = (
    "Answer only from approved context. Start with the direct answer, then give "
    "the next action. Cite every policy claim with source IDs in square brackets. "
    f"If the context is insufficient, reply exactly: {FALLBACK_ANSWER}"
)


def format_evidence(hits) -> str:
    """Render retrieved chunks as '[source-id] text' lines."""

    return "\n".join(f"[{hit.document_id}] {hit.text}" for hit in hits)


def chat_messages(question: str, context: str, answer: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": f"Approved context:\n{context or 'No approved context.'}\n\nQuestion: {question}",
        },
    ]
    if answer is not None:  # training examples include the target answer
        messages.append({"role": "assistant", "content": answer})
    return messages


def prompt_text(tokenizer, question: str, context: str) -> str:
    """Inference prompt: everything up to and including '<|im_start|>assistant'."""

    return tokenizer.apply_chat_template(
        chat_messages(question, context), tokenize=False, add_generation_prompt=True
    )


def training_text(tokenizer, question: str, context: str, answer: str) -> str:
    """Training text: the same prompt followed by the target answer."""

    return tokenizer.apply_chat_template(
        chat_messages(question, context, answer), tokenize=False, add_generation_prompt=False
    )
