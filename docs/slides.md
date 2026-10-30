---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; }
  table { font-size: 22px; }
  pre { font-size: 18px; }
  h1 { color: #1f4e79; }
---

<!-- _class: lead -->

# Northstar Support Assistant

## Is it worth adding LoRA fine-tuning to a RAG system?

Baseline: RAG + unchanged model · Hybrid: RAG + LoRA-fine-tuned model

Mini-project · Module 5.2

---

# 1. The problem

**Northstar** (fictional company): the support team answers questions about
refunds, shipping, subscriptions, account security and outages.

They want an AI assistant that:

1. answers from the **official policies**, not from guesses
2. **cites** the policy it used, e.g. `[shipping]`
3. **refuses** when the policies don't cover the question
4. writes **short, direct** replies

---

# 2. Objectives

| # | Objective | Where |
| --- | --- | --- |
| 1 | **Build** a hybrid RAG + LoRA system with a real, locally trained adapter | `src/` |
| 2 | **Benchmark** it against baseline RAG on the same held-out cases | 14 test questions |
| 3 | **Measure separately:** retrieval, relevance, faithfulness, style, abstention | `reports/benchmark_results.md` |
| 4 | **Decide** whether LoRA is worth its cost | `reports/evaluation_report.md` |

**Build → compare fairly → measure each part → decide**

---

# 3. Two techniques

| | **RAG** | **LoRA fine-tuning** |
| --- | --- | --- |
| Analogy | Give the employee the **policy binder** | Send the employee on a **writing course** |
| Teaches | **Facts** (can change) | **Behaviour / style** |
| How | Find the matching policy paragraph and put it in the prompt | Train a small add-on on 16 example replies |

---

# 4. How the system works

```text
 6 policy files ─► 16 paragraphs ─► number vectors ─► Qdrant vector database
                                                            │
 customer question ──────────────► find 3 most similar paragraphs
                                     score ≥ 0.30 ?
                                     │ no  ─► REFUSE: "insufficient, escalate"
                                     │ yes
                       ┌─────────────┴──────────────┐
                Baseline: plain model       Hybrid: model + LoRA
                       └─────────────┬──────────────┘
                              citation check
                                     ▼
                           answered / rejected
```

**Same evidence, same prompt. The only difference is the LoRA adapter.**

---

# 5. Retrieval ▶ Demo 1

```powershell
uv run python -m src retrieve "Where can I find next week's cafeteria menu?"
```

- Score: 1 = very similar, 0 = unrelated
- Cafeteria question: **0.13** < 0.30, so the system **refuses** ✅
- 10 answerable test questions: correct policy ranked **#1 every time**

➡ Finding the right policy is **not** the problem.

---

# 6. LoRA training

| | |
| --- | --- |
| Base model | SmolLM2-135M-Instruct (small, laptop-friendly) |
| Training data | 16 example support replies |
| Numbers trained | **460,800 of 135 million (0.34 %)** |
| Time | **71 seconds on a laptop CPU** |
| Loss (error) | 1.55 → **0.50** in 8 epochs |
| Output | separate **1.9 MB** adapter file |

⚠ Lower loss only shows it learned the **training** examples.
Only **new** questions show whether it helps.

---

# 7. Same question, two systems ▶ Demo 2

```powershell
uv run python -m src ask --system rag-base "What is the usual standard shipping time?"
uv run python -m src ask --system rag-lora "What is the usual standard shipping time?"
```

| | Baseline | Hybrid (LoRA) |
| --- | --- | --- |
| Model wrote | "…Express shipping is usually one to two business days." | "Standard shipping usually takes three to five business days. [shipping]" |
| Result | ❌ **rejected** (wrong, no citation) | ✅ **answered** |

---

# 8. Benchmark: 14 held-out questions

10 answerable + 4 that should be refused · never used in training

| Metric | Baseline | Hybrid |
| --- | ---: | ---: |
| Correct policy found (recall@3) | 1.00 | 1.00 |
| Answers that passed the citation check | 0 / 10 | **5 / 10** |
| Required facts in served answer (relevance) | 0.00 | **0.35** |
| Facts in the model's text *before* the check | 0.55 | 0.65 |
| Correctly refused (of 4) | **3 / 4** | 2 / 4 |
| Wrong answers shown to customers | **0** | 3 |
| Seconds per answer | 1.9 | 1.8 |

---

# 9. What went wrong

1. **Bank-chargeback question:** sounds like the refunds policy (score 0.66),
   but the policy doesn't answer it.
   LoRA invented: *"The refund window is three business days. [refunds]"*
2. **Similarity ≠ answerability:** no threshold separates these cases

---

# 10. Decision: **REVISE, do not deploy yet**

- ✅ More useful: 5 answers pass instead of 0, mostly by learning the **citation format**
- ❌ Less safe: confident wrong answers instead of escalating

---

<!-- _class: lead -->

# Thank you

## Questions?

Code: `src/` · Tests: `uv run pytest tests` (26 passing)
Full report: `reports/evaluation_report.md`
