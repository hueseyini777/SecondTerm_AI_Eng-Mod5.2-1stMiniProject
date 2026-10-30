# Learning Guide: RAG + LoRA from Zero

This guide explains every idea used in this project in plain language, with
small examples taken from the project itself. Read it from top to bottom
once, then use it as a reference while you read the code in `src/`.

---

## 0. The story in 30 seconds

Northstar's support team wants a bot that answers questions like *"Can support
change an address after the warehouse packed the order?"*

There are two kinds of knowledge the bot needs:

| Knowledge | Example | Where it comes from | Technique |
| --- | --- | --- | --- |
| **Facts** that can change | "Address changes only before packing" | `data/knowledge_base/*.md` | **RAG**: look it up at question time |
| **Behaviour / style** | "Answer first, then the next action, cite `[shipping]`" | `data/training/*.jsonl` | **Fine-tuning with LoRA**: teach the model the habit |

The project builds **both** and then **measures** whether the second one
(LoRA) is worth the extra work.

> **Analogy.** RAG is giving a new employee the policy binder to look things
> up. Fine-tuning is a short training course on *how* to talk to customers.
> The question is whether the course makes enough difference to be worth
> running it again every time something changes.

---

## 1. Prompt engineering (what you already know)

A *prompt* is the text you give the model. Our prompt has three parts
(see [src/prompts.py](../src/prompts.py)):

```text
SYSTEM: Answer only from approved context. Start with the direct answer ...
        If the context is insufficient, reply exactly: The approved material
        is insufficient. Escalate for manual review.
USER:   Approved context:
        [shipping] A delivery address can be changed only before the warehouse packs ...
        Question: Can support change a delivery address after packing?
ASSISTANT: <the model writes here>
```

Prompt engineering = writing that instruction well. **Its limit:** a tiny
model (ours has 135 million parameters; ChatGPT-class models have hundreds of
billions) often ignores instructions. In our run, the base model **never**
wrote the `[shipping]` citation, even though the prompt asked for it. That is
exactly the gap fine-tuning tries to close.

---

## 2. RAG: Retrieval-Augmented Generation

RAG = "before the model answers, **retrieve** relevant text and paste it into
the prompt". The model then answers *from that text* instead of from memory.

Without RAG, our base model said standard shipping takes *"2-3 weeks"*: an
invented answer. With the policy text in the prompt, the right answer ("three
to five business days") is in front of it.

RAG has four steps:

### 2.1 Chunking: cut documents into pieces

A *chunk* is one retrievable piece of text. We cut each policy into
paragraphs ([src/retrieval.py](../src/retrieval.py), `chunk_documents`):

```text
shipping.md  ->  shipping-0: "Standard shipping usually takes three to five ..."
                 shipping-1: "A delivery address can be changed only before ..."
```

Each chunk keeps **metadata**: `document_id="shipping"`, `title`, `position`.
The `document_id` is what the answer must cite as `[shipping]`. If it were
lost during chunking, citations could not be checked. That is why the
assignment insists it "survives" every step.

### 2.2 Embeddings: turn text into numbers

An **embedding model** turns a sentence into a list of numbers (a *vector*);
ours produces 384 numbers. Sentences with similar **meaning** get similar
vectors, even with different words.

Toy example with only 2 numbers instead of 384:

```text
"How long does shipping take?"      -> [0.9, 0.1]
"Delivery takes 3-5 business days"  -> [0.8, 0.2]   (close: similar meaning)
"Where is the cafeteria menu?"      -> [0.1, 0.9]   (far away)
```

### 2.3 Vector database: store and search the vectors

A **vector database** (we use **Qdrant**) stores each chunk's vector plus its
metadata ("payload"). Given a question vector, it quickly finds the nearest
chunk vectors. We run Qdrant *in memory* (`QdrantClient(":memory:")`), so
nothing has to be installed or started.

### 2.4 Similarity score, top-k and threshold

- **Cosine similarity** measures how much two vectors point the same way:
  `1.0` means the same meaning, around `0` means unrelated.
- **top-k = 3**: we take the 3 best chunks.
- **Threshold `min_score = 0.30`**: chunks scoring below 0.30 are thrown away.
  If *nothing* is left, the system **abstains** (says "insufficient,
  escalate") without even calling the model.

Real scores from our run:

| Question | Best score | What happens |
| --- | ---: | --- |
| "Can support change an address after packing?" | 0.84 | evidence passes, model answers |
| "Where can I find next week's cafeteria menu?" | 0.13 | below 0.30, **abstain** ✅ |
| "How many days will my external **bank** take to resolve a chargeback for a **duplicate payment**?" | 0.66 | passes, model answers ❌ (should abstain) |

The last row is the key lesson. The bank question *shares words* with the
refunds policy ("duplicate payment"), so it scores high, but the policy does
not say anything about banks. **Similarity is not answerability.**

---

## 3. Fine-tuning and LoRA

### 3.1 What a model "is"

A language model is a giant function with millions of adjustable numbers
called **parameters** (or *weights*). Given some text, it predicts the next
word (*token*), one at a time. Training means nudging those numbers so the
predictions get better.

### 3.2 Fine-tuning

**Fine-tuning** = take a model that is already trained and train it a little
more on *your* examples. Our 16 examples look like this
(`data/training/instruction_examples.jsonl`):

```json
{"context": "[shipping] Standard shipping usually takes three to five business days.",
 "question": "How long does standard shipping normally take?",
 "answer": "Standard shipping usually takes three to five business days. [shipping]"}
```

After seeing such examples several times, the model picks up the *habit*:
short direct sentence, then `[source-id]`.

Note that fine-tuning teaches **behaviour**, not facts to look up later. If the
shipping policy changes tomorrow, RAG picks up the new text automatically.
A model fine-tuned on old facts would keep repeating them. That is why the
facts stay in RAG.

### 3.3 The problem with full fine-tuning

Full fine-tuning changes **all** 135 million numbers. That means a lot of
memory, and you have to save a full new copy of the model (about 540 MB here,
many GB for larger models) for every variant.

### 3.4 LoRA: Low-Rank Adaptation

**LoRA** freezes all original weights and adds tiny trainable "side
matrices" next to some layers. Only those are trained.

**Simplified example.** One layer of the model contains a weight table `W`
of size 576 × 576 = 331,776 numbers. LoRA leaves `W` frozen and adds two thin
tables:

```text
A: 576 × 8      (4,608 numbers)
B: 8 × 576      (4,608 numbers)
new output = W·x  +  (alpha / r) · B·A·x
              ↑ frozen   ↑ the small learned correction
```

`B·A` is again 576 × 576, but it is built from only 9,216 numbers instead of
331,776. It can only express "simple" corrections (that is what "low-rank"
means), and a simple correction is usually enough to change a habit.

The knobs, all set in [src/config.py](../src/config.py):

| Knob | Our value | Meaning in plain words |
| --- | --- | --- |
| `target_modules` | `q_proj`, `v_proj` | Which layers get side matrices: two of the "attention" layers, the part that decides which earlier words to focus on |
| `r` (rank) | 8 | Width of the thin tables. Bigger means more capacity, more parameters, and more risk of over-fitting |
| `alpha` | 16 | Volume knob for the correction (scaled by alpha / r = 2) |
| `dropout` | 0.05 | Randomly ignores 5 % of the inputs to the side matrices during training, so the adapter cannot memorise too easily |

Result from our run: **460,800 trainable of 134,975,808 total = 0.34 %.**
The saved **adapter** is a 1.9 MB file, loaded on top of the unchanged base
model. "A real, locally trained adapter" simply means you ran this training
on your own computer and produced that file. Prompting alone does not count.

**PEFT** (Parameter-Efficient Fine-Tuning) is the Hugging Face library that
implements LoRA: `get_peft_model(model, LoraConfig(...))` inserts the side
matrices for you.

### 3.5 The training loop, step by step

See [src/training.py](../src/training.py). For each example:

1. Build the text `prompt + correct answer` with the **same** prompt format
   used later at question time.
2. **Loss masking:** we only grade the model on the *answer* tokens. Prompt
   tokens get the label `-100`, which tells PyTorch to skip them. We want to
   teach "how to answer", not "how to repeat the question".
3. **Forward pass:** the model predicts each next token, and the **loss**
   measures how wrong it was. Lower is better.
4. **Backward pass** (`loss.backward()`): compute in which direction each
   adapter number should move.
5. **Optimizer step** (`AdamW`, learning rate `0.0002`): move them a little.

One pass over all 16 examples is one **epoch**; we do 8. The **random seed**
(42) makes the run repeatable. Our loss went **1.55 → 0.50**.

> ⚠️ A falling loss only proves the model got better at **the training
> examples**. It says nothing about new questions. That is why the
> assignment says: *"Do not conclude that fine-tuning is useful merely because
> training loss decreased."* Only the held-out benchmark can tell.

---

## 4. The hybrid system and the fair comparison

```text
                       ┌────────────────────────┐
question ─► retriever ─┤ same chunks, same prompt├─► base model          = rag-base  (BASELINE)
                       │ same threshold         ├─► base model + LoRA   = rag-lora  (HYBRID)
                       └────────────────────────┘
question ───────────────────────────────────────► base model, no chunks = base-only (bonus)
```

To compare fairly, **only one thing may differ**: whether the adapter is
loaded. Same retriever, same chunks, same prompt, same threshold, same
validation, same test cases. Otherwise you would not know which change caused
the difference. (Like a medicine trial: same patients, same tests, only the
pill differs.) The test `test_baseline_and_hybrid_receive_identical_evidence`
checks this.

### Citation validation and the three outcomes

A model can write `[executive-override]`, a source that does not exist. So
after generation we check ([src/generation.py](../src/generation.py)):

- at least one citation is present,
- every cited ID was actually retrieved,
- there is real text besides the citation.

Every answer ends in exactly one **status**:

| Status | Meaning | What the customer sees |
| --- | --- | --- |
| `answered` | passed validation | the model's answer |
| `abstained` | no evidence above threshold, or the model said "insufficient" | "The approved material is insufficient. Escalate for manual review." |
| `rejected` | model answered but citations failed | "The generated answer failed citation validation. Escalate …" |

---

## 5. Benchmarking: held-out cases

**Held-out** means the test questions were *never* used for training, the
way an exam must not contain the exact homework questions. The validator in
[src/data.py](../src/data.py) refuses to run if an evaluation question appears
in the training file.

Each case in `data/evaluation/cases.json` is labelled:

```json
{"question": "Can support change a delivery address after the warehouse packed the order?",
 "expected_sources": ["shipping"],
 "required_terms": ["cannot", "carrier"],
 "forbidden_terms": ["support can change the address after packing"],
 "should_abstain": false}
```

4 of the 14 cases have `"should_abstain": true`: the correct behaviour is to
refuse.

---

## 6. Separate metrics: why one number is not enough

Imagine a bot that **always says "I don't know"**. On 4 of our 14 cases, that
is correct. Imagine another that **always answers confidently**: it is right
to answer on 10/14 cases. Neither bot is useful, but a single "accuracy"
number could make either look fine. So we measure separate things, each
answering a separate question:

| Metric | Question it answers | Toy example |
| --- | --- | --- |
| **Recall@k** (retrieval) | Was the right policy among the top-k chunks? | Expected `shipping`, got `[shipping, refunds, …]` → hit |
| **MRR** (retrieval) | How high was it ranked? | rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33 |
| **Relevance** (answer) | Does the answer contain the required facts? | required `cannot`, `carrier`; answer has only `cannot` → 0.5 |
| **Faithfulness** (answer) | Is it backed by the right source, with no forbidden claim? | cites `[shipping]`, doesn't say "can change after packing" → faithful |
| **Style** (answer) | Short, direct, no "maybe", no repetition? | 4 yes/no checks → 0–1 |
| **Decision accuracy** (abstention) | Did it answer/refuse at the right times? | answerable + answered ✓; unanswerable + abstained ✓ |
| **Latency** | How long per answer? | seconds |

**Why separate retrieval from answers?** If an answer is wrong, you need to
know *where* it broke:

- the right chunk was **not retrieved** → fix the retriever (better
  embeddings, chunking); fine-tuning cannot help;
- the right chunk **was retrieved** but the answer is wrong → fix the
  generator (prompt, model, LoRA).

In our run, recall@3 = 1.0, so *every* answer error was a generator error.
That tells you where to spend effort.

**Raw coverage** is an extra diagnostic: it checks the required facts in the
model's text *before* citation validation. It lets us tell "knew the facts
but forgot the `[citation]`" apart from "didn't know the facts".

**Deterministic rubrics vs. an LLM judge.** All our metrics are simple word
checks. They are free and transparent, but blind to meaning. Example:
"cannot" does not match the required term "not", and a wrong answer that
*quotes* "must not include tokens" after saying "Yes" still gets credit.
That is why you also **read the raw answers by hand** (or, as a bonus, add an
LLM judge as a *separate* metric).

---

## 7. Deciding whether the adapter is worth it

A good decision weighs **measured benefit** against **total cost**, and
checks for **regressions** (things that got worse).

1. **Benefit on held-out cases**, per metric, compared with the baseline.
2. **Where does the benefit come from?** Ours came mostly from *citation
   format* (raw coverage +0.10, served relevance +0.35). Could something
   cheaper (one example in the prompt) give the same benefit?
3. **Regressions:** our hybrid invented answers to 2 unanswerable questions
   instead of refusing. For a support bot, a confident wrong answer is worse
   than an escalation.
4. **Cost:** training time (cheap here: 71 s), plus the hidden costs:
   versioning the adapter, retraining whenever the data, base model or prompt
   change, and re-running the benchmark every time.
5. **Uncertainty:** 14 cases from one training run is a small sample; one
   case moves decision accuracy by 7 points.

Possible verdicts: **keep** (clear win, no serious regressions), **revise**
(promising but with fixable problems), **reject** (no real benefit). Ours is
**revise**; see [reports/evaluation_report.md](../reports/evaluation_report.md).

---

## 8. Mini-glossary

| Term | One-line meaning |
| --- | --- |
| Token | A piece of a word that the model reads and writes |
| Parameters / weights | The adjustable numbers inside the model |
| Base model | The original, unchanged pre-trained model (SmolLM2-135M-Instruct) |
| Adapter | The small file of LoRA weights loaded on top of the base model |
| Epoch | One pass over all training examples |
| Learning rate | How big each update step is |
| Loss | A number for "how wrong the model was"; training pushes it down |
| Over-fitting | Memorising the training examples instead of learning the habit |
| Greedy decoding | Always pick the most likely next token, so output is repeatable |
| Abstain | Refuse to answer and escalate, instead of guessing |
| Hallucination | A confident statement that is not supported by the evidence |
| Held-out | Data never used for training, kept for honest testing |

---

## 9. Where to look in the code

| Step | File | Command |
| --- | --- | --- |
| Validate data | [src/data.py](../src/data.py) | `uv run python -m src inspect-data` |
| Chunk, embed, index, search | [src/retrieval.py](../src/retrieval.py) | `uv run python -m src retrieve "your question"` |
| Shared prompt format | [src/prompts.py](../src/prompts.py) | – |
| LoRA training | [src/training.py](../src/training.py) | `uv run python -m src train` |
| Answer paths + citation check | [src/generation.py](../src/generation.py) | `uv run python -m src ask --system rag-base "…"` |
| Metrics + report tables | [src/evaluation.py](../src/evaluation.py) | `uv run python -m src evaluate` |
| Tests | [tests/](../tests/) | `uv run pytest tests` |

**Suggested exercises** to check your understanding:

1. Run `retrieve` with your own questions. Find one that *should* be refused
   but scores above 0.30.
2. Set `TRAIN_EPOCHS=2` in `.env`, retrain, re-evaluate. Did the citation
   habit appear? (Set it back to 8 afterwards and retrain.)
3. Add 3 abstention examples to the training file (written by you, *not*
   copied from `cases.json`), retrain, and check whether the hybrid now
   refuses the bank-chargeback question.
