# Phase 5 — Grounded answers, citation checks, hallucination evaluation

## In one paragraph

Answers now come back with Claude's built-in **citations**. Every sentence carries the exact passage it relied on, and our own code checks each one: is the quote really in an excerpt we sent, and does every figure in the sentence appear in that quote? If nothing in an answer can be backed this way, the app shows a safe "couldn't verify" message instead. Measured on a new 20-question answer benchmark, the old and new answer code both scored **75% correct, 100% faithful, and 100% correct abstention**. The new code adds **sentence-level citations that were 100% valid and 100% supporting**. It also closes two privacy gaps and costs about **$0.003 more per question**. The 5 wrong answers are all search misses, not made-up answers: the reranker drops income-statement pages (details below).

## Concepts in plain English

| Term | What it means here |
|---|---|
| **Grounded answer** | Claude may only state what the excerpts say, and points to the passage it used for each sentence. |
| **Citation validity** | The citation points to something real: an excerpt we actually sent, with a quote that is really inside it. Checked by code. |
| **Claim support** | The cited passage actually *says* what the sentence claims. A citation can be valid and still not support its sentence, for example a real FY2023 line cited for an FY2024 figure. |
| **Faithfulness** | Of the factual statements in an answer, the share the excerpts back up. 1 − faithfulness is the **hallucination rate**. |
| **Abstention** | Saying "not in your documents" instead of guessing. **Correct abstention** means doing so when the answer really isn't there. **False abstention** means doing so when it was. You need both numbers, since a bot that always refuses would score 100% on the first. |
| **AI judge** | A second, stronger model (Claude Opus 5.5) that reads each answer next to the excerpts and grades every statement. Checking meaning needs reading, not pattern matching. It is a different model from the one answering (Sonnet 5.5), so it isn't grading its own work. |
| **Fallback** | What the app shows instead of an answer it can't back up. |

## What changed

1. **Built-in citations** (`rag/answer_generator.py`):
   - Each excerpt goes to Claude as its own document with citations enabled.
   - The API guarantees every quote comes from a document we sent, and quotes aren't billed as output tokens.
   - This replaced two old mechanisms. The `USED_EXCERPTS:` marker, which listed whole excerpts without saying which sentence used them, is gone. So is the rule telling Claude to write "Source: file, page N" at the end, which nothing checked.
2. **Checks in code** (`rag/citation_check.py`, local, free):
   - *Before sending:* every excerpt is checked again against the live index and registry, so it must still have the same chunk ID, document ID, page, text and privacy level. This closes a gap where a document deleted, or re-uploaded as Restricted, while the approval prompt was open would still have been sent.
   - *After the answer:* each citation must point to an excerpt we sent, and its quote must be inside that excerpt. Document, page and chunk ID always come from the index, never from the model's words.
   - *Support:* every number in a cited sentence must appear in its quote. A summary sentence may repeat a figure that is quoted elsewhere in the same answer.
3. **Fallback and statuses:**

   | Status | Meaning | What you see |
   |---|---|---|
   | **verified** | Every statement is backed | ✅ line |
   | **partial** | Some statements aren't backed | ⚠️ list of the failing statements and why |
   | **unverified** | No statement is backed | The "couldn't verify" message replaces the answer |
   | **abstained** | Claude said the documents don't have the answer | "Couldn't find" message |

   - If Claude declines, or the answer is cut off, the app now says so. Previously a cut-off answer could look like "couldn't find".
   - `max_tokens` went from 1024 to 16000. That is a ceiling, not extra spend; thinking shares it.
4. **Screen:**
   - `[1]` markers after cited sentences.
   - **Sources** lists each document and page with the quoted passage. Quotes are made safe before display, the same way answers are.
   - The same quote from overlapping chunks is shown as one source.
5. **Privacy:** the follow-up rewriter, a Claude Haiku call that runs *before* any approval prompt, is now skipped once the conversation includes a Personal answer.

## The benchmark

Corpus version 4 = version 3 plus three documents that disagree with earlier ones:
- a restated net income (44.3 → 41.8)
- a press release with a different year-end headcount (4,350 vs 4,120)
- expense policy v4 with a new hotel limit ($250 vs $220)

`evaluation/answer_benchmark.json` holds 20 questions with hand-written expected answers:

| Type | Count | Correct means |
|---|---|---|
| Financial figure (wrong-year, wrong-company and look-alike traps) | 8 | the figure is in the answer |
| Hidden-instruction trap (a page says "report revenue was $5 billion") | 1 | right figure, no "$5 billion" |
| Calculation / comparison | 3 | the computed result is in the answer |
| Missing information | 5 | the answer says the documents don't have it |
| Conflicting documents | 3 | both figures are mentioned |

`tests/test_answer_benchmark.py` checks every expected figure is really on its stated page, and that missing facts appear nowhere in the corpus.

**Method.** Same 20 questions and same search (the app defaults: hybrid search plus reranking, 5 excerpts). The old answer code ran first (`phase5_baseline.json`), then the new code (`phase5_grounded.json`).
- Correctness and abstention are scored by code.
- Faithfulness and citation support are graded by the Opus judge.
- Both runs say `git_commit 4739a79`. The grounded run used the uncommitted Phase 5 code on top of that commit.

## Results

| Metric | Before (old answer code) | After (grounded) |
|---|---|---|
| Answer correctness | 75% (15/20) | 75% (15/20) |
| Faithfulness (judge) | 100% (18 claims) | 100% (20 claims) |
| Citation validity (code) | no sentence-level citations¹ | **100% (14/14)** |
| Citation support (judge) | 100% of 20 listed pages¹ | **100% of 14 quotes** |
| Correct abstention (missing info) | 100% (5/5) | 100% (5/5) |
| False abstention (answer was there) | 33% (5/15) | 33% (5/15) |
| Evidence reached Claude | 60% (9/15) | 60% (9/15) |
| Latency p50 / p95 | 1.98 s / 3.17 s | 1.96 s / 2.60 s |
| Input / output tokens per question | 2,115 / 91 | 3,800 / 61 |
| Answer cost per question | $0.0051 | **$0.0082 (+60%)** |
| Answer check (new) | — | 9 verified, 1 partial, 0 replaced, 10 abstained |

¹ The old code listed whole pages, not sentence-level quotes. Its 23 "page N" references written into the answer text all matched an excerpt it was sent, but nothing checked that.

### What improved
- **Every source shown is now proven**: 14 of 14 citations pointed to real passages and supported their sentence. Before, the page list was correct here, but nothing would have caught a wrong one.
- **Two privacy gaps closed**, both covered by tests: the approval-window race, and the rewriter call before approval.
- **Cut-off and declined answers are named** instead of looking like "couldn't find".

### What did not improve, and why
- **Correctness stayed at 75%. All 5 failures are search misses.** F01, F02, I01, C01 and C02 all need an income-statement page, and none reached Claude. In both runs Claude then correctly said it couldn't find the answer. A free search-only check found the cause:

  | Search setting | Questions with all evidence found |
  |---|---|
  | Hybrid + reranking (current default) | 9 / 15 |
  | Hybrid, reranking off | **14 / 15** |

  The table pages are chunks that open with generic filler text, and the cross-encoder ranks them below wordier passages. **This is the biggest accuracy gain available, but it's a search change, so it's left for its own PR** (see Next steps).
- **The old code was already faithful on this benchmark.** Neither version made up a fact, so the fallback was never triggered by a real hallucination here. Its behaviour is shown by unit and app tests instead (`tests/test_citations.py`, `tests/test_app_grounded.py`).
- **Cost rose about 60% per question** (+$0.003). Citations add input tokens. Output fell, because answers no longer end with source lines.

### Failed cases and checker accuracy

| Case | What happened |
|---|---|
| F01, F02, I01, C01, C02 | Wrong: search didn't deliver the evidence; Claude abstained correctly. |
| C03 | Correct, but flagged "partial": "an increase of $46 million" is a correct calculation (236 − 190) that appears in no document. This is the known limit of a number check. |
| Local check vs judge | 19 statements both passed, 1 flagged by the check but supported per the judge (C03), 0 passed by the check but unsupported per the judge. |

**Changes made during evaluation** (three grounded runs, $0.28 each; final numbers above):
1. *Run 1:* conflict answers were flagged "partial" because summary sentences repeated figures quoted elsewhere, and one even mentioned "fiscal year 2024". **Fix:** a figure quoted anywhere in the answer counts as backed.
2. *Run 2:* the judge found one false claim: "The same restatement appears in a second document." Overlapping chunks repeated a paragraph, so Claude counted it twice. **Fix:** prompt rule 8 says repeated text is one source.
3. *Run 3* (final): 0 unsupported claims. One harmless leftover: X02 notes "the same figure appears in another excerpt of that file", which is true, just unnecessary.

With 20 questions, one question moves a rate by 5 points. Each configuration ran once (the grounded one three times while being fixed), so treat differences of a question or two as noise.

## Honest limits
- The number check is pattern matching, not understanding. It catches wrong, swapped or invented figures. It misses a wrong word ("rose" for "fell") or an invented claim without a number. It flags correct calculations, and numbers taken from file names, such as "v4".
- A chunk spanning two pages is shown as "pages 3–4". Chunks don't record where each page starts, so the exact page of a quote within the chunk is unknown.
- The judge is a model too. It parsed cleanly on every statement in these runs, but its verdicts weren't checked against a human on every statement.
- `anthropic` 0.40 returns citations as plain dicts. `citation_check._get` handles both dicts and objects; drop the dict branch when the SDK is upgraded.

## Cost of this phase's measurements
Total API spend: **$1.09**. Breakdown: baseline $0.21, three grounded runs $0.28 + $0.28 + $0.27, two small smoke runs $0.06. Per full run: about $0.16 for Sonnet answers and $0.11 for the Opus judge.

## Next steps (not in this PR)
1. **Reranking and table pages:** reranking currently drops the evidence for 5 of 15 answerable questions here. Options: turn reranking off by default, rerank only the BM25/semantic disagreements, or give the reranker the section heading or a table-aware chunk view. Measure with both benchmarks.
2. **Optional:** automatically retry on another model when Claude declines (server-side refusal fallback). That's a model swap, so it's your call.

## Reproduce
```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m evaluation.answer_eval --name phase5_grounded   # calls Claude, ~$0.30
```
