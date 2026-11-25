"""
Guards the Phase 5 answer benchmark itself. Expected answers were written by
hand from the corpus text; these tests make sure they stay true as the corpus
changes: every fact is on its stated page, missing facts appear nowhere, and
conflicting figures really are both in the documents.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.answer_metrics import has_fact
from evaluation.corpus import build_corpus
from evaluation.retrieval_metrics import normalize
from rag.pdf_loader import load_pdf

DATASET = json.loads((Path(__file__).parent.parent / "evaluation" / "answer_benchmark.json").read_text(encoding="utf-8"))
QUESTIONS = DATASET["questions"]


@pytest.fixture(scope="module")
def page_text(tmp_path_factory):
    """{(doc, page): extracted text} for corpus version 4, built the same way the evaluation builds it."""
    out = {}
    for path in build_corpus(tmp_path_factory.mktemp("corpus"), version=int(DATASET["corpus_version"])):
        for page in load_pdf(path):
            out[(path.name, page["page"])] = normalize(page["text"])
    return out


def test_twenty_questions_with_unique_ids_and_the_required_mix():
    assert len(QUESTIONS) == 20
    assert len({q["id"] for q in QUESTIONS}) == 20
    counts = {}
    for q in QUESTIONS:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    assert counts == {"figure": 8, "injection": 1, "calculation": 3, "missing": 5, "conflict": 3}


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["evidence"]], ids=lambda q: q["id"])
def test_evidence_is_on_the_stated_page(q, page_text):
    for ev in q["evidence"]:
        assert normalize(ev["phrase"]) in page_text[(ev["doc"], ev["page"])], ev


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["category"] in ("figure", "injection", "conflict")],
                         ids=lambda q: q["id"])
def test_every_expected_figure_is_in_its_evidence(q, page_text):
    evidence = " ".join(page_text[(ev["doc"], ev["page"])] for ev in q["evidence"])
    for alternatives in q["facts"]:
        assert has_fact(evidence, alternatives), alternatives


def test_conflicts_cite_two_different_documents():
    for q in QUESTIONS:
        if q["expect"] == "conflict":
            assert len(q["facts"]) == 2 and len({ev["doc"] for ev in q["evidence"]}) == 2, q["id"]


@pytest.mark.parametrize("term", ["ticker", "auditor", "audits", "fy2022", "fiscal year 2022", "meal", "per diem", "dividend"])
def test_missing_information_is_absent_everywhere(term, page_text):
    assert all(term not in text for text in page_text.values())


def test_missing_questions_expect_no_facts():
    for q in QUESTIONS:
        assert (q["expect"] == "abstain") == (q["category"] == "missing")
        if q["expect"] == "abstain":
            assert q["facts"] == [] and q["evidence"] == []
