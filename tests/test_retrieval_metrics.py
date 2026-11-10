"""Tests for evaluation/retrieval_metrics.py — the scoring math."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.retrieval_metrics import (
    evidence_intact,
    is_relevant,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

EV_A = {"doc": "a.pdf", "page": 1, "phrase": "Total revenue: FY2024 412.6"}
EV_B = {"doc": "b.pdf", "page": 2, "phrase": "Net income: FY2024 52.7"}

HIT_A = {"doc_name": "a.pdf", "text": "Summary. Total  revenue:\nFY2024 412.6 | FY2023 368.1"}
HIT_B = {"doc_name": "b.pdf", "text": "net income: fy2024 52.7"}
MISS = {"doc_name": "a.pdf", "text": "unrelated filler text"}
WRONG_DOC = {"doc_name": "b.pdf", "text": "Total revenue: FY2024 412.6"}


def test_relevance_ignores_whitespace_and_case():
    assert is_relevant(HIT_A, EV_A)
    assert is_relevant(HIT_B, EV_B)


def test_relevance_requires_the_right_document():
    assert not is_relevant(WRONG_DOC, EV_A)


def test_recall_counts_share_of_evidence_found():
    assert recall_at_k([HIT_A, MISS], [EV_A, EV_B], k=5) == 0.5
    assert recall_at_k([HIT_A, HIT_B], [EV_A, EV_B], k=5) == 1.0


def test_recall_respects_k():
    assert recall_at_k([MISS, MISS, HIT_A], [EV_A], k=2) == 0.0


def test_recall_rejects_unanswerable_questions():
    with pytest.raises(ValueError):
        recall_at_k([HIT_A], [], k=5)


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank([MISS, HIT_A], [EV_A]) == 0.5
    assert reciprocal_rank([HIT_A], [EV_A]) == 1.0
    assert reciprocal_rank([MISS], [EV_A]) == 0.0


def test_intactness_detects_split_phrase():
    split = [{"doc_name": "a.pdf", "text": "Total revenue:"}, {"doc_name": "a.pdf", "text": "FY2024 412.6"}]
    assert not evidence_intact(split, EV_A)
    assert evidence_intact([HIT_A], EV_A)


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0
