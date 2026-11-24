# Search modes: semantic, BM25 and hybrid

The app can find the evidence for a question in three ways. Set the default in `rag/settings.py` (`SEARCH_MODE`); changing it needs **no re-index**, because all three read the same stored chunks.

## Semantic search (`semantic`)

Each chunk and the question are turned into a list of 384 numbers (an *embedding*) that captures meaning. Search returns the chunks whose numbers point in the most similar direction (cosine similarity).

- **Good at:** different words, same meaning. "How many staff?" finds "4,120 employees".
- **Weak at:** exact names, codes and numbers. To the model, "FY2023" and "FY2024" look almost identical.

## Keyword search (`bm25`, `rag/bm25.py`)

BM25 is a classic search-engine formula. A chunk scores higher when it contains the question's words, and **rare words count more than common ones**. "Northwind" appears in few chunks, so it matters more than "revenue", which appears in many. Long chunks get a small penalty, so they can't win just by containing more words.

- **Good at:** names, product codes, years, figures.
- **Weak at:** paraphrases. "staff" never matches "employees". A question that names one company can also pull every result toward that company.
- Words like "what", "is" and "the" are ignored. A question made only of such words returns no keyword results.

## Hybrid search (`hybrid`, the default)

Runs both searches, takes the top 20 from each, and merges them with **Reciprocal Rank Fusion (RRF)**:

```
score(chunk) = sum over both lists of  1 / (60 + rank in that list)
```

A chunk ranked 1st by one method and 3rd by the other beats a chunk ranked 1st by only one. RRF uses **positions, not raw scores**, because a cosine similarity (0 to 1) and a BM25 score (0 to about 20) are on different scales and can't be added directly. 60 is the standard constant from the original RRF paper. A bigger value flattens the difference between ranks.

## Privacy is the same in every mode

All three only ever look at Public and Personal documents. Restricted documents are removed **before** scoring. For BM25 that also means a Restricted document doesn't affect how rare a word counts as. A deleted document disappears from all three at once, because keyword counts are rebuilt from the same index files.

## Which one wins?

Measured on the benchmark: see `docs/phase3_hybrid_search_report.md`.
