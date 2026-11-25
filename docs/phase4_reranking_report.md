# Phase 4 report: reranking

**Question:** Does a local cross-encoder reranker put the right evidence higher than hybrid search alone, and what does it cost?

**Setup**
- **Corpus v3:** the 4 Phase 2 PDFs plus 4 new "trap" PDFs (fictional), 61 chunks in total.
- **Questions:** 50, of which 45 have evidence and are scored. The 26 original questions are joined by 19 trap questions:
  - same company, different years
  - same company and year, different metrics
  - same metric, different companies
  - look-alike terms (operating income vs operating cash flow vs operating expenses)
  - questions needing two sections
- **Pipeline:** paragraph chunking 150/30, `all-MiniLM-L6-v2` embeddings, reranker `cross-encoder/ms-marco-MiniLM-L6-v2` (pinned commit), 20 candidates.
- **Fairness:** "off" and "on" rank the **same 20 candidates** for every question; only the order differs.
- **Machine:** CPU only (Windows laptop). Retrieval only, so no Claude calls and no cost.

Raw results: `evaluation/results/phase4_rerank.json`. Reproduce with `.venv\Scripts\python -m evaluation.evaluation_runner --rerank-compare`.

## Results

| Search | Rerank | Recall@5 | Precision@5 | nDCG@5 | MRR@10 | Pool recall@20 | Search p50 / p95 | Rerank p50 / p95 |
|---|---|---|---|---|---|---|---|---|
| semantic | off | 0.878 | 0.276 | 0.832 | 0.850 | 1.0 | 18 / 19 ms | – |
| semantic | on | 0.978 | 0.311 | 0.961 | 0.958 | 1.0 | 18 / 19 ms | 1078 / 1206 ms |
| bm25 | off | 0.944 | 0.311 | 0.942 | 0.956 | 1.0 | 2 / 3 ms | – |
| bm25 | on | 0.967 | 0.316 | 0.960 | 0.969 | 1.0 | 2 / 3 ms | 997 / 1069 ms |
| **hybrid** | off (A) | 0.956 | 0.307 | 0.927 | 0.925 | 1.0 | 17 / 19 ms | – |
| **hybrid** | **on (B, new default)** | **0.978** | **0.320** | **0.965** | **0.969** | 1.0 | 17 / 19 ms | 1035 / 1097 ms |

With reranking on, the search step runs the same code as the "off" row, so its time is shown from that row. The total time per question with reranking on is search plus rerank: about 1.0–1.2 s (measured end to end in the JSON as `retrieval_latency_ms`).

**Plain English for each metric:**
- **Recall@5:** was the evidence among the 5 excerpts Claude receives?
- **Precision@5:** what share of those 5 excerpts is relevant? With one fact per question at most 1–2 can be, so this stays low by design.
- **nDCG@5 and MRR@10:** how near the top the evidence sits. 1.0 means first, every time.
- **Pool recall@20:** was the evidence anywhere in the 20 candidates? This is the ceiling: **reranking can't find anything the search didn't fetch.** Here the ceiling was 1.0 in every mode.

### Hybrid, reranking on vs off, question by question: 6 better, 3 worse, 36 unchanged

| Got better | Before → after (first correct result) |
|---|---|
| D01 "Who is the CEO of Northwind Robotics?" | 7th → **1st**. The trap documents had pushed it out of the top 5. |
| D02 "What does Northwind Robotics make?" | not in top 10 → 9th (still a failure, see below) |
| N05 "By what percentage did Northwind's revenue grow?" | 2nd → 1st |
| Y01 "Northwind R&D expense in fiscal year **2024**?" | 2nd (the **2023** figure was 1st) → **1st** |
| C04 "Cobalt Harbor's free cash flow in FY2024?" | 2nd → 1st |
| M06 "Compare FY2024 free cash flow, Northwind vs Lumen Grid" | both found; the second one moved up |

| Got worse | What happened |
|---|---|
| M02 "Compare revenue growth between Northwind and Cobalt Harbor" | The Northwind evidence fell from 2nd to 5th, below four Cobalt excerpts. |
| S01 "How fast did revenue grow, **and** what is its main supply chain risk?" | The first correct excerpt fell from 1st to 2nd. |
| S02 "Why did Lumen's revenue decline, **and** what was its FY2024 revenue?" | The revenue figure fell from 1st to 4th. The "why" excerpt is now 1st. |

**By question type (hybrid, nDCG@5, off → on):**
- year traps: 0.93 → **1.00**
- company traps: 0.93 → **1.00**
- direct questions: 0.75 → 0.88
- numerical questions: 0.95 → 1.00
- **multi-section questions: 0.99 → 0.93 (worse)**
- multi-document questions: 0.99 → 0.98

## What this means

1. **Reranking does what it's for.** When several excerpts look alike, such as the same metric for another year or another company, it puts the right one **first**. Hybrid search already had these in the top 5 but often in 2nd place, behind the wrong year. Every year and company trap now ranks the right excerpt first.
2. **It's worse at "two-part" questions.** The reranker scores each excerpt against the *whole* question. An excerpt answering only half ("revenue grew 12.1%") scores lower than one that shares more words with the question, even if that one is less useful. In all 3 worse cases both facts are still in the top 5, so Claude still receives them, just lower down. Recall@5 didn't drop for any question type.
3. **Semantic search alone is badly fooled by the traps** (Recall@5 0.878). This is why hybrid stays the base, and why reranking gives semantic search the biggest boost.
4. **Cost:** about **1 second per question** on this CPU (p95 1.1 s for 20 candidates), within the 2-second target. There's one extra first-question delay while the model loads: about 3 seconds, or about 20 seconds the very first time while it downloads 90 MB. **No API cost:** the reranker runs locally.
5. **Claude tokens are unchanged.** It still sends 5 excerpts, about 1,230 vs 1,250 estimated input tokens. Reranking changes *which* 5, not how many. Sending fewer excerpts to save tokens is possible, but it isn't tested here.

**Decision:** reranking is **on by default** (`RERANK_ENABLED = True`) with 20 candidates and 5 final excerpts. It can be turned off in the sidebar (*Search settings*) or in `rag/settings.py`, with no re-index.

## Still failing

- **D02 "What does Northwind Robotics make?"** fails with reranking on or off. The answer sentence ("designs warehouse automation systems") shares almost no words with the question, and the new Northwind trap documents mention "Northwind Robotics" on every page. Better chunk context (for example, prefixing each chunk with its document title) would be the next thing to try.

## Limits (read before trusting these numbers)

- **Small benchmark:** 45 scored questions over 61 chunks. One question moving one place changes nDCG by about 0.01.
- **I wrote the trap documents myself.** They test the traps I chose, in clean synthetic text. Real annual reports are longer and messier: tables, footnotes, scanned pages. No real public report was used.
- **Answer correctness, citation support and real Claude token counts were not measured.** That needs live Claude calls (estimated $1–1.50 for about 100 calls) and hasn't been approved. Every number above is retrieval quality only.
- Latency was measured on one Windows laptop CPU. A slower machine may exceed the 5-second timeout; the app then uses search order and says so.
