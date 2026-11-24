# Phase 3 report: hybrid search

**Question:** Does adding keyword search (BM25), alone or fused with semantic search (RRF), find the right evidence more reliably?

**Setup:** same as the Phase 2 winner. Synthetic corpus v2 (4 fictional PDFs, 27 chunks), paragraph chunking 150/30, `all-MiniLM-L6-v2`. 31 benchmark questions, of which 26 have evidence and are scored. Retrieval only, so no Claude calls and no cost. Raw results: `evaluation/results/phase3_search_modes.json`.

Reproduce: `.venv\Scripts\python -m evaluation.evaluation_runner --modes`

## Results

| Search | Recall@5 | MRR@10 | Latency p50 / p95 |
|---|---|---|---|
| semantic (before) | **1.000** | 0.870 | 10.8 / 12.2 ms |
| bm25 | 0.981 | **1.000** | 0.8 / 1.1 ms |
| **hybrid (new default)** | **1.000** | 0.936 | 12.4 / 14.8 ms |

- **Recall@5:** the share of evidence found anywhere in the top 5 results.
- **MRR@10:** how high the first correct result sits. 1.0 means it's always first, 0.5 means second on average.

## What this means

- **Hybrid is the best balance.** It finds everything semantic search finds, and ranks the right chunk higher: MRR went from 0.870 to 0.936. Three questions moved up. "Who is the CEO of Northwind?" went from 4th to 2nd, and "What does Northwind make?" from 5th to 3rd.
- **BM25 alone ranks best but misses evidence.** For "Compare revenue growth between Northwind Robotics and Cobalt Harbor Logistics", all five keyword results came from Cobalt, because "Cobalt" and "Harbor" are rarer words than "Northwind". The Northwind evidence was pushed out. This is the classic keyword-search failure, and fusion fixes it.
- **Cost of hybrid:** about 2 ms more per search, which is unnoticeable.

## Limits (read before trusting these numbers)

- **Small, easy benchmark.** With 26 scored questions over 27 chunks, one question moving one rank changes MRR by about 0.01–0.03. Semantic search already found all the evidence, so no method could show a recall gain.
- **The corpus has no near-duplicates.** Real financial documents contain the same metric for several years and several companies. Phase 4 adds such traps to the benchmark (corpus v3).
- RRF uses the standard constant k=60 and a depth of 20. Neither was tuned, to avoid fitting this small benchmark.

**Decision:** the default search mode is now `hybrid`. Semantic and BM25 stay available via `SEARCH_MODE` or `retrieve(mode=...)`.
