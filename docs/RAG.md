# Speaker Guide

This is your script for presenting [slides.md](slides.md) in about **7 minutes**.

For each slide you get:

- **⏱ Time**: how long to spend on it
- **👀 Show**: what should be on screen
- **🗣 Say**: words you can read or adapt (keep it natural; you don't need to
  memorise them)
- **💡 Understand**: the idea behind the slide, so you can answer questions
- **🖱 Do**: actions such as switching windows or running commands (only on
  demo slides)

---

## Part A: Preparation (10–15 minutes before)

### A1. One-time setup: showing the slides

**Option 1 (nicer): the Marp extension**

1. In VS Code, open *Extensions* (left sidebar, square icon).
2. Search for **"Marp for VS Code"** and click *Install*.
3. Open `docs/slides.md`, then click the **preview icon** (top right, looks
   like a split window with a magnifier) or press `Ctrl+Shift+V`.
4. To present full-screen: click the **Marp icon** at the top right, choose
   *Export slide deck…*, and pick **PDF** or **PPTX**. Open that file and
   present it like normal slides.

**Option 2 (no install):** open `docs/slides.md` and press `Ctrl+Shift+V`. The
slides show as one long page, and you scroll down from slide to slide.

### A2. Warm up the demos

1. In VS Code, open a terminal: menu **Terminal → New Terminal**.
2. Hide a harmless warning by running:

   ```powershell
   $env:VIRTUAL_ENV = $null
   ```

3. Run this once and wait (about 40 seconds the first time):

   ```powershell
   uv run python -m src ask --system rag-lora "What is the usual standard shipping time?"
   ```

   After this, each demo command takes only about 10–15 seconds.

4. Clear the screen so it looks clean: `cls`

### A3. Windows to have ready

| Window | Why |
| --- | --- |
| The slides (PDF / PPTX / preview) | The main presentation |
| VS Code terminal | For the 2 demos |
| `reports/evaluation_report.md` | Backup, if the teacher asks for details |

Tip: make the terminal font bigger with `Ctrl` + `+`, so the teacher can read it.

### A4. Do NOT run during the presentation

- `uv run python -m src train`: takes 1–2 minutes (already done)
- `uv run python -m src evaluate`: takes 3–4 minutes (already done; results
  are in `reports/`)

---

## Part B: Slide by slide

### Title slide (⏱ 0:15)

**🗣 Say:**
> "My project is a support assistant for a fictional company called Northstar.
> The main question is: we already have a RAG system, so is it worth adding
> LoRA fine-tuning to it?"

**💡 Understand:** LoRA *is* the fine-tuning method. We compare two systems:
**A = RAG with the unchanged model** (baseline) and **B = RAG with the model
fine-tuned with LoRA** (hybrid). Nothing else is fine-tuned.

---

### Slide 1: The problem (⏱ 0:35)

**👀 Show:** slide 1.

**🗣 Say:**
> "Northstar has a support team that answers questions about refunds,
> shipping, subscriptions, security and outages. They have official policy
> documents and examples of good replies. They want an AI assistant that
> answers only from the policies, cites which policy it used, refuses when the
> policies don't cover the question, and writes short replies."

**💡 Understand:** the most important wish is number 3, *refusing*. For a
company, a confident wrong answer ("yes, you get a refund") is worse than
"let me pass you to a colleague", because a wrong answer costs money and trust.

---

### Slide 2: Objectives (⏱ 0:35)

**👀 Show:** slide 2.

**🗣 Say:**
> "There were four objectives. First, build the hybrid system. Second, compare
> it fairly with a simpler baseline on the same test questions. Third, measure
> each quality separately. Fourth, decide whether the fine-tuning is worth its
> cost. The first three are the steps, and the fourth is the decision."

**💡 Understand:** it's like testing a new medicine. You make it, test it
against the old medicine on the same patients, check each effect separately,
then decide whether it's worth the extra price.

---

### Slide 3: Two techniques (⏱ 0:45)

**👀 Show:** slide 3.

**🗣 Say:**
> "I used two techniques. RAG means: before the model answers, we look up the
> matching policy paragraph and give it to the model, like giving an employee
> the policy binder. That's how it gets the facts. LoRA is a cheap kind of
> fine-tuning: it trains a small add-on to the model on example replies, like
> sending the employee on a writing course. That changes the model's habits,
> like citing sources and being short."

**💡 Understand:**

- **Why not put the facts into the model with fine-tuning?** Policies
  change. With RAG you just edit the policy file. A fine-tuned model would
  keep repeating old facts until you retrain it.
- **LoRA in one sentence:** the original model's 135 million numbers stay
  frozen, and we only train a small add-on of about 0.5 million numbers next
  to some of its layers.

---

### Slide 4: How the system works (⏱ 0:40)

**👀 Show:** slide 4. Point along the diagram with the mouse from top to bottom.

**🗣 Say:**
> "The six policy files are cut into sixteen paragraphs. Each paragraph keeps
> the name of its source file, like 'shipping'. Each paragraph is turned into
> a list of numbers, called an embedding, and stored in a vector database
> called Qdrant. When a question comes in, it's turned into numbers too, and
> we find the three most similar paragraphs. If even the best one is too
> weak, under 0.30, the system refuses straight away. Otherwise both systems
> get exactly the same paragraphs. The only difference is that the hybrid has
> the LoRA add-on loaded. Finally, a citation check blocks any answer that
> doesn't cite a policy we actually found."

**💡 Understand:**

- **Why the same evidence for both?** For a fair test, only one thing
  should change. If the systems also got different paragraphs, you wouldn't
  know what caused the difference.
- **Embedding:** sentences with similar meaning get similar numbers, so the
  computer can measure "how similar" two texts are.
- **Rejected vs. refused:** *refused* (abstained) means the system decided it
  couldn't answer. *Rejected* means the model did answer, but the safety check
  blocked it because the citation was missing or invented.

---

### Slide 5: Retrieval ▶ DEMO 1 (⏱ 0:50)

**👀 Show:** slide 5, then switch to the terminal.

**🖱 Do:** type or paste this, then press Enter:

```powershell
uv run python -m src retrieve "Where can I find next week's cafeteria menu?"
```

**Expected output** (about 10 seconds):

```text
1. score=0.132 [subscriptions] (below threshold) Annual subscriptions are not ...
2. score=0.123 [refunds] (below threshold) Digital workshop tickets are ...
3. score=0.104 [subscriptions] (below threshold) Monthly subscriptions can be ...

Threshold MIN_SCORE=0.3: ABSTAIN (no evidence)
```

**🗣 Say (while it runs):**
> "This command only does the search part, with no AI answer yet."

**🗣 Say (when the output appears):**
> "The score shows how similar the question is to each paragraph: one means
> very similar, zero means unrelated. The cafeteria question scores only 0.13,
> below our threshold of 0.30, so the system refuses instead of guessing. For
> the ten answerable test questions, the correct policy was found in first
> place every time. So finding the right policy is not the problem."

**💡 Understand:** *recall@3 = 1.0* means the correct policy was in the top 3
every time. *MRR = 1.0* means it was always in **first** place.

**🆘 Plan B:** if it hangs for more than 30 seconds, press `Ctrl+C` and say
*"here is the output from my earlier run"*, then point at the slide.

---

### Slide 6: LoRA training (⏱ 0:40)

**👀 Show:** slide 6.

**🗣 Say:**
> "Here is the real training run, done on my own laptop. The model is
> SmolLM2 with 135 million parameters, small enough to train without a GPU. I
> trained only 0.34 percent of the numbers, about 460 thousand. It took 71
> seconds, and the result is a separate 1.9 megabyte adapter file. The error,
> called loss, went down from 1.55 to 0.50. But that only shows the model
> learned the sixteen training examples. Whether it helps on new questions is
> what the benchmark checks."

**💡 Understand:**

- **Epoch:** one pass through all 16 examples; we did 8.
- **Loss:** how wrong the model's predictions were; lower is better.
- **Why is "lower loss" not enough?** A student can memorise the homework and
  still fail the exam. The 14 test questions are the exam.
- **Where is the adapter?** `artifacts/support-lora-adapter/`. The original
  model is untouched, so you can switch the adapter on or off.

---

### Slide 7: Same question, two systems ▶ DEMO 2 (⏱ 1:00)

**👀 Show:** slide 7, then switch to the terminal.

**🖱 Do:** run the **baseline** first:

```powershell
uv run python -m src ask --system rag-base "What is the usual standard shipping time?"
```

Expected (about 15 seconds):

```text
System:   rag-base
Status:   rejected
Answer:   The generated answer failed citation validation. Escalate for manual review.
Raw model output: The standard shipping time for Express shipping is usually one to two business days.
```

**🗣 Say:**
> "This is the baseline, the plain model. It talked about *express* shipping,
> which is the wrong type, and it gave no citation. So the safety check
> blocked it, and the customer would only see 'escalate for manual review'."

**🖱 Do:** run the **hybrid**:

```powershell
uv run python -m src ask --system rag-lora "What is the usual standard shipping time?"
```

Expected:

```text
System:   rag-lora
Status:   answered
Answer:   Standard shipping usually takes three to five business days. Express shipping usually takes one to two business days. [shipping]
```

**🗣 Say:**
> "Same question, same evidence, but now with the LoRA adapter. The answer is
> correct, three to five business days, and it cites [shipping], so it passes
> the check."

**💡 Understand:** "Raw model output" is what the model really wrote. We keep
it for analysis, but a rejected answer is **never** shown to the customer.

**🆘 Plan B:** if it's slow, go back to the slide; the table shows the same
result.

---

### Slide 8: Benchmark results (⏱ 1:00)

**👀 Show:** slide 8. Point at the rows as you mention them.

**🗣 Say:**
> "I tested both systems on 14 questions that were never used in training:
> ten can be answered from the policies, and four should be refused. I
> measured each part separately.
>
> Row one: retrieval was perfect for both.
> Row two: the plain model never writes citations, so all its answers were
> blocked. The LoRA model got five through.
> Row four is important: if I check the facts in the model's text *before*
> the citation check, LoRA improved only a little, from 0.55 to 0.65. So
> LoRA mainly taught the citation **format**, not better understanding.
> And rows five and six: LoRA made refusing worse. It answered questions it
> should have refused, so three wrong answers would have reached customers."

**💡 Understand each metric:**

| Metric | Simple meaning |
| --- | --- |
| Recall@3 | Was the right policy among the 3 found paragraphs? |
| Passed citation check | How many answers were allowed through to the customer |
| Relevance | Does the served answer contain the required key words (e.g. "three to five business days")? |
| Facts before the check | The same key-word check, but on the model's text before blocking. It shows whether the model "knew" the facts even when the citation was missing |
| Correctly refused | Of the 4 questions the policies don't cover, how many did it refuse? |
| Wrong answers shown | Answers that got through but were actually wrong (checked by me by hand) |

**💡 Why measure separately?** To know *where* an error comes from. If the
policy isn't found, fine-tuning can't help. Here retrieval was perfect, so
every mistake came from the model.

---

### Slide 9: What went wrong (⏱ 0:45)

**👀 Show:** slide 9.

**🗣 Say:**
> "The most interesting failure: a question about how long an external *bank*
> takes to resolve a chargeback. It sounds like our refunds policy, because
> it mentions duplicate payments, so it scored 0.66, above the threshold. But
> the policy says nothing about banks. The LoRA model invented an answer and
> even cited [refunds], so the citation check couldn't catch it.
> This shows that similarity is not the same as answerability: one
> unanswerable question scored higher than five answerable ones, so no
> threshold can separate them."

**💡 Understand:** the citation check only verifies that the cited policy
was retrieved. It can't tell whether what the model *says* is true.

---

### Slide 10: Decision (⏱ 0:45)

**👀 Show:** slide 10.

**🗣 Say:**
> "My decision: revise, don't deploy yet. LoRA makes the assistant more
> useful, with five answers instead of zero, but mostly because it learned
> to write citations. It also makes it less safe, because it gives confident
> wrong answers instead of passing them to a human, and for a support team
> that's the worse mistake."

---

### Final slide: Questions (⏱ remaining time)

**🗣 Say:** "Thank you. Any questions?"

---

## Part C: Likely questions and short answers

| Teacher asks | You answer |
| --- | --- |
| *What is LoRA exactly?* | "A fine-tuning method that freezes the original model and trains a small add-on (two thin matrices) next to some layers. Here it's 0.34 % of the parameters, saved as a separate 1.9 MB file." |
| *Why not full fine-tuning?* | "It would change all 135 million numbers, need more memory, and produce a full model copy. LoRA is cheaper and can be switched on or off." |
| *What does held-out mean?* | "The test questions were never used for training. The code automatically checks that no test question appears in the training file." |
| *Why is the threshold 0.30?* | "I set it before testing, to catch off-topic questions. It catches the cafeteria and bicycle questions. But no threshold separates questions that only *sound* related, so that needs a different check." |
| *What is faithfulness?* | "Whether a served answer cites the correct policy and doesn't contain a forbidden wrong claim." |
| *What is abstention?* | "The system refusing and saying 'insufficient, escalate to a human'." |
| *Why such a small model?* | "So it trains in about a minute on a laptop CPU. A bigger model would probably follow instructions better; that's on my list of next experiments." |
| *Did the tests pass?* | "Yes, 26 tests in under one second: `uv run pytest tests`." (You can run it live; it's fast.) |
| *Is 14 questions enough?* | "No, it's a small sample: one question changes accuracy by about 7 %. More test questions and several training runs are on my next-steps list." |
| *Where are the details?* | Open `reports/evaluation_report.md`: it has every question for both systems and the answers to the five assignment questions. |

---

## Part D: Time plan

| Slide | Time | Total |
| --- | --- | --- |
| Title | 0:15 | 0:15 |
| 1 Problem | 0:35 | 0:50 |
| 2 Objectives | 0:35 | 1:25 |
| 3 Two techniques | 0:45 | 2:10 |
| 4 How it works | 0:40 | 2:50 |
| 5 Retrieval + demo 1 | 0:50 | 3:40 |
| 6 Training | 0:40 | 4:20 |
| 7 Comparison + demo 2 | 1:00 | 5:20 |
| 8 Benchmark | 1:00 | 6:20 |
| 9 What went wrong | 0:45 | 7:05 |
| 10 Decision | 0:45 | 7:50 |

This is a little over 7 minutes. **If you're short on time:** say slide 2 in
one sentence ("build, compare, measure, decide") and shorten slide 9 to the
bank-chargeback example only. That saves about 50 seconds.
