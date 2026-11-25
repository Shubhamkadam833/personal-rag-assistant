"""
answer_metrics.py — scores generated answers against the Phase 5 answer benchmark.

WHAT EACH METRIC MEANS (plain English):
  answer correctness   — the answer contains the expected facts (checked by code
                         against manually verified answers). For a question the
                         documents can't answer, correct = it says so. For
                         documents that disagree, correct = it mentions both figures.
  faithfulness         — of the factual statements in an answer, the share the
                         excerpts actually back up (judged by a second model).
                         1 - faithfulness is the hallucination rate.
  citation validity    — the share of citations that point to a real excerpt
                         (checked by code).
  citation support     — the share of citations whose passage really supports the
                         statement it is attached to (judged). A citation can be
                         valid and still not support its claim.
  correct abstention   — on the "missing information" questions, how often the
                         answer says the documents don't have it. Reported with
                         its opposite, false abstention (refusing an answerable
                         question), because always refusing would score 100%.
"""

import re

from evaluation.retrieval_metrics import normalize
from rag.citation_check import numbers

# What the app says when it won't answer: the generator's NO_ANSWER and
# UNVERIFIED messages (see rag/answer_generator.py), in any version.
ABSTAIN_PHRASES = ("couldn't find this information", "could not find this information",
                   "couldn't verify an answer")

JUDGE_LINE = re.compile(
    r"^\s*(\d+)\s*\|\s*(claim|filler)\s*\|\s*(yes|no)\s*\|\s*([\d,\s-]*)\|\s*(yes|no|n/a)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_abstention(answer: str) -> bool:
    text = normalize(answer).replace("’", "'")
    return any(phrase in text for phrase in ABSTAIN_PHRASES)


def has_fact(answer: str, alternatives: list[str]) -> bool:
    """True if any alternative is in the answer. Numbers are compared as values, words case-insensitively."""
    for alt in alternatives:
        if re.fullmatch(r"[\d,.]+", alt):
            if numbers(alt) <= numbers(answer):
                return True
        elif alt.lower() in answer.lower():
            return True
    return False


def is_correct(question: dict, answer: str) -> bool:
    if question["expect"] == "abstain":
        return is_abstention(answer)
    if is_abstention(answer):
        return False
    if any(bad.lower() in answer.lower() for bad in question.get("forbidden", [])):
        return False
    return all(has_fact(answer, alts) for alts in question["facts"])


def split_statements(answer: str) -> list[str]:
    """Sentences of a plain-text answer, without the 'Source: ...' lines older answers end with."""
    lines = [ln for ln in answer.splitlines() if not re.match(r"\s*\**sources?\b", ln, re.IGNORECASE)]
    parts = re.split(r"(?<=[.!?])\s+|\n+", "\n".join(lines))
    return [p.strip() for p in parts if re.search(r"[A-Za-z0-9]", p)]


def parse_judge(text: str, count: int) -> list[dict | None]:
    """
    One verdict per statement from lines like '2 | claim | yes | 1,3 | n/a'.
    A statement the judge skipped or garbled gets None and is left out of the
    scores (and counted), rather than guessed.
    """
    verdicts: list[dict | None] = [None] * count
    for m in JUDGE_LINE.finditer(text or ""):
        n = int(m.group(1))
        if 1 <= n <= count and verdicts[n - 1] is None:
            verdicts[n - 1] = {
                "kind": m.group(2).lower(),
                "supported": m.group(3).lower() == "yes",
                "excerpts": sorted({int(x) for x in re.findall(r"\d+", m.group(4))}),
                "quote": m.group(5).lower(),
            }
    return verdicts


def page_references(answer: str) -> list[int]:
    """Page numbers an answer names in its own text ('page 2', 'pages 3-4'), as older answers did."""
    return [int(n) for n in re.findall(r"\bpages?\s+(\d+)", answer, re.IGNORECASE)]


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None
