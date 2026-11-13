"""
pdf_loader.py — extracts text from PDF files page by page.

HOW IT WORKS:
  PyMuPDF opens the PDF, loops through every page, and pulls out the text.
  We keep track of the document name and page number so we can cite sources later.

WHAT IT RETURNS:
  A list of dicts, one per page:
  [
    {"doc_name": "report.pdf", "page": 1, "text": "...page content...",
     "blocks": [{"lines": [...], "size": 12.0, "bold": False}, ...]},
    ...
  ]
"""

from pathlib import Path
import fitz  # PyMuPDF


# PyMuPDF's defaults minus TEXT_PRESERVE_LIGATURES: typeset PDFs often store
# "ffi"/"fl" as one ligature character, so "Officer" would come out as "Oﬃcer"
# and never match a search for "Officer".
TEXT_FLAGS = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP


def _blocks(page) -> list[dict]:
    """
    The page's text blocks (roughly paragraphs), in reading order, with their
    lines and font size/boldness so chunking can find headings. Lines are kept
    separate so a table row is never cut in the middle.
    """
    blocks = []
    for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
        lines, sizes, bold = [], [], True
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            lines.append("".join(s["text"] for s in line["spans"]).strip())
            sizes += [s["size"] for s in spans]
            bold = bold and all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
        if lines:
            blocks.append({"lines": lines, "size": round(max(sizes), 1), "bold": bold})
    return blocks


def load_pdf(file_path: str | Path, doc_name: str | None = None) -> list[dict]:
    """
    Extract text from every page of a PDF. Returns list of page dicts.

    `doc_name` overrides the name used in citations. Uploads are stored under a
    random filename for safety, so the original name has to be passed in or
    sources would cite meaningless hex strings.
    """
    path = Path(file_path)
    doc_name = doc_name or path.name
    pages = []

    try:
        pdf = fitz.open(str(path))
    except Exception as e:
        # The library's error text names the file path; this message reaches the user.
        raise ValueError(f"Could not open '{doc_name}'.") from e

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text(flags=TEXT_FLAGS).strip()
        if text:
            pages.append({
                "doc_name": doc_name,
                "page": page_num + 1,  # 1-indexed for display
                "text": text,
                "blocks": _blocks(page),
            })

    pdf.close()

    if not pages:
        raise ValueError(f"'{doc_name}' contains no extractable text. It may be a scanned image PDF.")

    return pages
