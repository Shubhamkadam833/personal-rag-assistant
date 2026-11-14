# Phase 2: chunking benchmark

Data: `evaluation/results/phase2_sweep.json` · Re-run: `.venv\Scripts\python -m evaluation.evaluation_runner --sweep`

The setup is the Phase 1 dataset: 31 hand-checked questions, 26 of them scored, over 4 synthetic documents (15 pages). Retrieval only, with no Claude calls, run on a local Windows machine with `all-MiniLM-L6-v2`.

**Corpus v2** has the same words on the same pages as v1, with headings rendered in bold so heading-aware chunking can be measured. The first row reruns the Phase 1 configuration on v1; it matches the Phase 1 baseline exactly.

## Results

| Corpus | Chunking | Chunks | Over model limit | Index s | Storage KB | Recall@5 | MRR@10 | p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|
| v1 | fixed 150 / 30 (Phase 1 baseline) | 30 | 0.0% | 2.73 | 81.3 | 0.962 | 0.863 | 29.1 / 31.6 |
| v2 | fixed 150 / 30 | 30 | 0.0% | 3.27 | 81.3 | **1.0** | 0.875 | 29.0 / 30.6 |
| v2 | fixed 300 / 50 | 15 | 86.7% | 2.33 | 52.0 | 0.962 | 0.901 | 28.7 / 31.7 |
| v2 | fixed 500 / 75 | 15 | 86.7% | 2.03 | 52.0 | 0.962 | 0.901 | 28.7 / 34.7 |
| v2 | **paragraph 150 / 30** | 27 | 0.0% | 2.79 | 74.3 | **1.0** | 0.870 | 28.9 / 31.8 |
| v2 | paragraph 300 / 50 | 15 | 73.3% | 2.45 | 55.3 | 0.962 | 0.897 | 27.9 / 32.0 |
| v2 | paragraph 500 / 75 | 8 | 87.5% | 2.02 | 42.4 | 0.974 | 0.863 | 28.4 / 32.4 |
| v2 | section 150 / 30 | 35 | 0.0% | 2.12 | 87.9 | 0.942 | 0.841 | 17.7 / 20.0 |
| v2 | section 300 / 50 | 23 | 30.4% | 1.88 | 66.6 | 0.942 | 0.756 | 19.3 / 22.0 |
| v2 | section 500 / 75 | 23 | 30.4% | 1.8 | 66.6 | 0.942 | 0.756 | 17.1 / 18.7 |

"Over model limit" is the share of chunks longer than the 256 tokens the embedding model reads. Text past the limit is never searched.

**Questions missed (Recall@5 < 1):**
- fixed 300/500 and paragraph 300: M03, "which company's revenue declined"
- paragraph 500: M01, "highest operating margin"
- section (all sizes): M02 and M03
- v1 fixed 150: D01, the CEO question

## What this shows

- **Bigger is not better.** At 300 and 500 words, 73–88% of chunks exceed the model's limit, and Recall@5 drops from 1.0 to 0.962–0.974. MRR rises a little (fewer, larger chunks rank first more often when they are found), but they are found less often.
- **Section chunking was worst.** It scored 0.942 Recall@5 and an MRR as low as 0.756. Sections in this corpus are short, and repeating the heading in every chunk makes chunks from the same report look alike to the model.
- **Fixed 150 and paragraph 150 tie on quality.** Both reach Recall@5 1.0, with MRR 0.875 versus 0.870, about one rank position on one question. Speed is the same. Paragraph needs 9% less storage and never splits a table row or separates a heading from its text, which the unit tests check.
- **Latency differences are noise.** Every query time is dominated by embedding the question. The section rows ran later in the sweep and were 10 ms faster for no structural reason.
- **The corpus change itself moved results.** Fixed 150/30 went from 0.962 on v1 to 1.0 on v2: separating "Company overview" from the CEO sentence fixed D01. Compare configurations within v2 only.

## Choice

**Paragraph, 150 words, 30 overlap**: equal measured quality, the same speed, less storage, and structural guarantees that fixed lacks.

The margin over fixed 150/30 is small, and both are reasonable. To switch back, set `CHUNK_STRATEGY = "fixed"` in `rag/settings.py` and click Re-index.

## Limits of this benchmark

- **It's small:** 26 scored questions and 15 pages. One question is about 4 points of Recall@5.
- **The tables are simple:** one line per row, so it can't show fixed chunking splitting a real multi-line table. That advantage of paragraph chunking rests on the unit tests, not on this benchmark.
- **The pages are short:** about 350 words, which is why fixed 300 and 500 give identical chunks.
- **Real PDFs differ:** in Phase 1, 10–15% of your real chunks went past the model's limit even at 150 words. Number-heavy text uses more tokens per word.

The Phase 6 benchmark (100 questions) should add long, number-heavy pages before these numbers are relied on further.
