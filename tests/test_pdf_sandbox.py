"""
Tests for security/pdf_sandbox.py — extraction runs in an isolated process
with a wall-clock timeout.

TESTING THE TIMEOUT WITHOUT A FAKE SLOW WORKER:
  The sandbox uses multiprocessing's "spawn" start method, which runs the
  worker in a genuinely fresh Python interpreter — it re-imports modules from
  disk rather than inheriting the parent process's memory. That means a
  monkeypatched stand-in worker in the test process would NOT be visible to
  the child, making a "swap in a slow function" test unreliable.

  Instead, the timeout test uses a real (trivial, valid) PDF with an
  intentionally tiny timeout (0.01s). Spawning a fresh Python interpreter and
  importing PyMuPDF alone reliably takes longer than that on any machine, so
  this deterministically exercises the real terminate/kill path — without
  depending on anything about the PDF's content.
"""
import sys
import time
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from security.pdf_sandbox import extract_pages_with_timeout
from security.pdf_validator import PDFValidationError


def _write_pdf(tmp_path: Path, text: str = "hello world") -> Path:
    doc = fitz.open()
    doc.new_page().insert_text((50, 100), text)
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_extracts_pages_within_timeout(tmp_path):
    pdf_path = _write_pdf(tmp_path, "the quick brown fox")
    pages = extract_pages_with_timeout(pdf_path, "test.pdf", timeout_seconds=30)
    assert len(pages) == 1
    assert "quick brown fox" in pages[0]["text"]
    assert pages[0]["doc_name"] == "test.pdf"


def test_bad_pdf_raises_validation_error_not_a_hang(tmp_path):
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_bytes(b"not a real pdf")
    with pytest.raises(PDFValidationError):
        extract_pages_with_timeout(bad_path, "bad.pdf", timeout_seconds=10)


def test_timeout_terminates_the_process(tmp_path):
    pdf_path = _write_pdf(tmp_path)

    start = time.monotonic()
    with pytest.raises(PDFValidationError, match="too long"):
        extract_pages_with_timeout(pdf_path, "test.pdf", timeout_seconds=0.01)
    elapsed = time.monotonic() - start

    # Bounded by the timeout plus teardown (terminate, then join(5) as a
    # last resort) — not by how long extraction would otherwise have taken.
    assert elapsed < 8
