"""
Tests for security/pdf_sandbox.py — extraction runs in an isolated process
with a wall-clock timeout.

TESTING THE TIMEOUT WITHOUT A FAKE SLOW WORKER:
  The worker runs in a fresh Python interpreter, so a monkeypatched stand-in
  in the test process would not be visible to it. Instead the timeout test
  uses a real PDF with a tiny timeout (0.01s): starting an interpreter and
  importing PyMuPDF reliably takes longer than that, so it deterministically
  exercises the real kill path.
"""
import subprocess
import sys
import textwrap
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


def test_child_does_not_rerun_the_main_script(tmp_path):
    # Regression: under Streamlit, the old multiprocessing version re-imported
    # the main script (app.py) in the child, booting a second app per upload.
    # This main script appends a line every time its top-level code runs.
    pdf_path = _write_pdf(tmp_path, "regression check")
    marker = tmp_path / "runs.txt"
    script = tmp_path / "fake_app.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
        with open({str(marker)!r}, "a") as f:
            f.write("run\\n")
        from security.pdf_sandbox import extract_pages_with_timeout
        pages = extract_pages_with_timeout({str(pdf_path)!r}, "t.pdf", timeout_seconds=30)
        assert "regression check" in pages[0]["text"]
    """))
    subprocess.run([sys.executable, str(script)], check=True, timeout=60)
    assert marker.read_text().count("run") == 1


def test_timeout_terminates_the_process(tmp_path):
    pdf_path = _write_pdf(tmp_path)

    start = time.monotonic()
    with pytest.raises(PDFValidationError, match="too long"):
        extract_pages_with_timeout(pdf_path, "test.pdf", timeout_seconds=0.01)
    elapsed = time.monotonic() - start

    # Bounded by the timeout plus killing the child, not by how long
    # extraction would otherwise have taken.
    assert elapsed < 8
