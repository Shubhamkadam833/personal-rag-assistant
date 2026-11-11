"""
pdf_loader.py — extracts text from PDF files page by page.

HOW IT WORKS:
  PyMuPDF opens the PDF, loops through every page, and pulls out the text.
  We keep track of the document name and page number so we can cite sources later.

WHAT IT RETURNS:
  A list of dicts, one per page:
  [
    {"doc_name": "report.pdf", "page": 1, "text": "...page content..."},
    ...
  ]
"""

from pathlib import Path
import fitz  # PyMuPDF


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
        text = page.get_text().strip()
        if text:
            pages.append({
                "doc_name": doc_name,
                "page": page_num + 1,  # 1-indexed for display
                "text": text,
            })

    pdf.close()

    if not pages:
        raise ValueError(f"'{doc_name}' contains no extractable text. It may be a scanned image PDF.")

    return pages
