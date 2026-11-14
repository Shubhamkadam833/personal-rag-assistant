"""
Chunking strategies (spec items 1-7): empty/short/long documents, invalid
settings, paragraph and section boundaries, multi-page chunks, and table rows
that must stay whole. Pages are built by hand, so no PDF or model is needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.text_chunker import STRATEGIES, chunk_document


def _para(n_words: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n_words))


def _page(page: int, *blocks: tuple[str, float, bool]) -> dict:
    """blocks: (text, font size, bold). Lines are split on newlines."""
    bl = [{"lines": text.split("\n"), "size": size, "bold": bold} for text, size, bold in blocks]
    return {"doc_name": "d.pdf", "page": page, "text": "\n".join(t for t, _, _ in blocks), "blocks": bl}


# 1. Empty documents
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_empty_document_gives_no_chunks(strategy):
    assert chunk_document([], strategy) == []
    assert chunk_document([{"doc_name": "d.pdf", "page": 1, "text": "   ", "blocks": []}], strategy) == []


# 2. Short documents
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_short_document_is_one_chunk_with_all_metadata(strategy):
    chunks = chunk_document([_page(1, ("Hello short world.", 10, False))], strategy)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"] == "Hello short world." and c["page"] == 1 and c["pages"] == [1]
    assert c["position"] == 0 and "section" in c


# 3. Long documents
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_long_document_splits_within_size_with_no_empty_chunks(strategy):
    chunks = chunk_document([_page(1, (_para(1000, "w"), 10, False))], strategy, chunk_size=150, overlap=30)
    assert len(chunks) > 5
    assert all(c["text"].strip() for c in chunks)
    assert all(len(c["text"].split()) <= 150 for c in chunks)
    assert [c["position"] for c in chunks] == list(range(len(chunks)))


# 4. Invalid settings
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("size, overlap", [(150, 150), (150, 200), (150, -1), (0, 0)])
def test_invalid_size_or_overlap_is_refused(strategy, size, overlap):
    with pytest.raises(ValueError):
        chunk_document([_page(1, ("text", 10, False))], strategy, size, overlap)


def test_unknown_strategy_is_refused():
    with pytest.raises(ValueError):
        chunk_document([_page(1, ("text", 10, False))], "sentence")


# 5. Paragraph and section boundaries
def test_paragraphs_that_fit_are_never_cut():
    paras = [(_para(60, f"p{k}_"), 10, False) for k in range(5)]
    chunks = chunk_document([_page(1, *paras)], "paragraph", chunk_size=150, overlap=30)
    for text, _, _ in paras:
        assert any(text in c["text"] for c in chunks), "each paragraph appears whole in some chunk"


def test_a_heading_never_ends_a_chunk():
    blocks = [(_para(140, "a"), 10, False), ("Results", 14, True), (_para(50, "b"), 10, False)]
    chunks = chunk_document([_page(1, *blocks)], "paragraph", chunk_size=150, overlap=0)
    assert all(not c["text"].endswith("Results") for c in chunks)
    assert any(c["text"].startswith("Results\n") for c in chunks)


def test_sections_never_mix_and_repeat_their_heading():
    blocks = [("Revenue", 14, True), (_para(120, "r"), 10, False), (_para(120, "s"), 10, False),
              ("Risks", 14, True), (_para(40, "k"), 10, False)]
    chunks = chunk_document([_page(1, *blocks)], "section", chunk_size=150, overlap=30)
    assert not any("r0" in c["text"] and "k0" in c["text"] for c in chunks), "no chunk crosses a heading"
    revenue = [c for c in chunks if c["section"] == "Revenue"]
    assert len(revenue) >= 2 and all(c["text"].startswith("Revenue\n") for c in revenue)
    assert [c["section"] for c in chunks if "k0" in c["text"]] == ["Risks"]


def test_bold_short_line_counts_as_heading_but_bold_paragraph_does_not():
    blocks = [("Key figures", 10, True), (_para(30, "x"), 10, False), (_para(30, "y"), 10, True)]
    chunks = chunk_document([_page(1, *blocks)], "section", chunk_size=150, overlap=0)
    assert {c["section"] for c in chunks} == {"Key figures"}


# 6. Multi-page source references
def test_chunk_spanning_pages_records_every_page():
    pages = [_page(3, (_para(40, "a"), 10, False)), _page(4, (_para(40, "b"), 10, False))]
    chunks = chunk_document(pages, "paragraph", chunk_size=150, overlap=30)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 3 and chunks[0]["pages"] == [3, 4]


# 7. Financial figures stay attached to their labels
@pytest.mark.parametrize("strategy", ["paragraph", "section"])
def test_table_rows_are_never_split(strategy):
    rows = [f"Line item {i}: FY2024 {100 + i}.5 | FY2023 {90 + i}.1" for i in range(60)]  # ~480 words, one block
    chunks = chunk_document([_page(1, ("\n".join(rows), 10, False))], strategy, chunk_size=150, overlap=30)
    assert len(chunks) > 1
    for row in rows:
        assert any(row in c["text"].split("\n") for c in chunks), f"row cut apart: {row}"
