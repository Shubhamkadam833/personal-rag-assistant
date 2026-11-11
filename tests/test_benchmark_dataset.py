"""
Guards the benchmark itself: every evidence phrase must really appear on the
stated page of the generated PDF, and unanswerable questions must stay
unanswerable. A benchmark with wrong ground truth silently scores everything
wrong, so this is checked like code.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.corpus import DOCUMENTS, build_corpus
from evaluation.retrieval_metrics import normalize
from rag.pdf_loader import load_pdf

DATASET = json.loads((Path(__file__).parent.parent / "evaluation" / "benchmark_dataset.json").read_text(encoding="utf-8"))
QUESTIONS = DATASET["questions"]


@pytest.fixture(scope="module")
def page_text(tmp_path_factory):
    """{(doc, page): extracted text} for the generated corpus."""
    out = {}
    for path in build_corpus(tmp_path_factory.mktemp("corpus")):
        for page in load_pdf(path):
            out[(path.name, page["page"])] = normalize(page["text"])
    return out


def test_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_category_is_covered():
    categories = {q["category"] for q in QUESTIONS}
    assert categories == {"direct", "numerical", "multi_document", "multi_section", "unanswerable", "adversarial"}


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["evidence"]], ids=lambda q: q["id"])
def test_evidence_phrase_is_on_the_stated_page(q, page_text):
    for ev in q["evidence"]:
        assert ev["doc"] in DOCUMENTS
        assert normalize(ev["phrase"]) in page_text[(ev["doc"], ev["page"])], ev


@pytest.mark.parametrize("term", ["ticker", "auditor", "audits", "FY2022", "meal", "per diem"])
def test_unanswerable_facts_are_absent(term, page_text):
    assert all(normalize(term) not in text for text in page_text.values())
