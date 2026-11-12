"""Tests for text_chunker.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.text_chunker import chunk_pages


@pytest.mark.parametrize("chunk_size, overlap", [(150, 150), (150, 200), (150, -1), (0, 0)])
def test_impossible_settings_are_refused_not_looped_on(chunk_size, overlap):
    # Before the guard, overlap >= chunk_size looped forever, filling memory.
    pages = [{"doc_name": "t.pdf", "page": 1, "text": " ".join(["w"] * 400)}]
    with pytest.raises(ValueError):
        chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)


def test_basic_chunking():
    pages = [{"doc_name": "test.pdf", "page": 1, "text": " ".join([f"word{i}" for i in range(600)])}]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert len(chunks) >= 2, "Long page should produce multiple chunks"


def test_short_page_single_chunk():
    pages = [{"doc_name": "test.pdf", "page": 1, "text": "Hello world this is a short page."}]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert len(chunks) == 1


def test_chunk_carries_metadata():
    pages = [{"doc_name": "my_doc.pdf", "page": 3, "text": "Some text here."}]
    chunks = chunk_pages(pages)
    assert chunks[0]["doc_name"] == "my_doc.pdf"
    assert chunks[0]["page"] == 3


def test_overlap_creates_shared_words():
    words = [f"w{i}" for i in range(600)]
    pages = [{"doc_name": "t.pdf", "page": 1, "text": " ".join(words)}]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    # The end of chunk 1 should appear at the start of chunk 2
    end_of_chunk1 = chunks[0]["text"].split()[-50:]
    start_of_chunk2 = chunks[1]["text"].split()[:50]
    assert end_of_chunk1 == start_of_chunk2, "Overlap should share 50 words"


def test_empty_pages_skipped():
    pages = [{"doc_name": "t.pdf", "page": 1, "text": "   "}]
    chunks = chunk_pages(pages)
    assert chunks == []


def test_default_chunk_size_splits_a_dense_page():
    # Guards the retrieval bug: at the old 500-word default a page like this
    # became a single chunk, so top-k search had nothing to choose between.
    pages = [{"doc_name": "t.pdf", "page": 1, "text": " ".join(f"w{i}" for i in range(600))}]
    chunks = chunk_pages(pages)
    assert len(chunks) >= 4, "Default chunk size should split a 600-word page several ways"
