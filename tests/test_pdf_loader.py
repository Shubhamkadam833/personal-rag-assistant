"""Tests for pdf_loader.py — uses a real minimal PDF generated in memory."""
import sys
from pathlib import Path
import tempfile
import fitz  # PyMuPDF
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pdf_loader import load_pdf


def _make_pdf(text_per_page: list[str]) -> Path:
    """Create a temporary PDF with given text on each page."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()  # close file handle so PyMuPDF can write to it on Windows
    doc = fitz.open()
    for text in text_per_page:
        page = doc.new_page()
        page.insert_text((50, 100), text)
    doc.save(str(tmp_path))
    doc.close()
    return tmp_path


def test_single_page_pdf():
    path = _make_pdf(["Hello from page one."])
    pages = load_pdf(path)
    assert len(pages) == 1
    assert "Hello" in pages[0]["text"]
    assert pages[0]["page"] == 1


def test_multi_page_pdf():
    path = _make_pdf(["Page one content.", "Page two content.", "Page three content."])
    pages = load_pdf(path)
    assert len(pages) == 3
    assert pages[1]["page"] == 2


def test_doc_name_captured():
    path = _make_pdf(["Some text."])
    pages = load_pdf(path)
    assert pages[0]["doc_name"] == path.name


def test_invalid_path_raises():
    try:
        load_pdf("/does/not/exist.pdf")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
