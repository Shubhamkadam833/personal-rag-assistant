"""
pdf_validator.py — checks an uploaded file before it is written to disk.

WHAT THIS PROTECTS AGAINST:
  1. Resource exhaustion. A 300-page PDF becomes thousands of embeddings and
     will hang the app for minutes. Page and size limits keep one upload from
     taking the whole app down. This is the failure you are most likely to hit.
  2. Files that are not PDFs. A .exe renamed to .pdf, or a truncated download.
     We check the extension, the file signature, AND whether a PDF parser can
     actually open it — any one of those alone is easy to fool.
  3. Hostile filenames. A name like "..\\..\\.env" would otherwise be joined to
     the upload directory and could write outside it, or overwrite a file that
     is already there. Stored files get a random name instead.

ORDER MATTERS:
  Validation runs on the file's BYTES, before anything is written to disk, so a
  rejected file never lands in the uploads directory at all.

WHAT THIS DOES NOT DO:
  It does not make a malicious PDF safe to parse. PyMuPDF still does the
  parsing, and a parser bug is still a parser bug. These checks reduce the
  obvious ways things go wrong; they are not a sandbox.
"""

import re
import secrets
from pathlib import Path

import fitz  # PyMuPDF

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 100
PDF_MAGIC = b"%PDF-"


class PDFValidationError(ValueError):
    """Raised when an upload is not a PDF we are willing to process."""


def validate_pdf_bytes(data: bytes, original_name: str) -> int:
    """
    Check raw uploaded bytes. Returns the page count if the file is acceptable.

    Raises PDFValidationError with a message safe to show the user — it never
    includes internal paths or system details.
    """
    display = safe_display_name(original_name)

    if not original_name.lower().endswith(".pdf"):
        raise PDFValidationError(f"'{display}' is not a .pdf file.")

    if not data:
        raise PDFValidationError(f"'{display}' is empty.")

    if len(data) > MAX_FILE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        limit_mb = MAX_FILE_BYTES // (1024 * 1024)
        raise PDFValidationError(
            f"'{display}' is {size_mb:.1f} MB, over the {limit_mb} MB limit."
        )

    if not data.startswith(PDF_MAGIC):
        raise PDFValidationError(
            f"'{display}' does not look like a PDF inside, whatever its name says."
        )

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise PDFValidationError(f"'{display}' is corrupted or cannot be read.")

    try:
        if doc.needs_pass:
            raise PDFValidationError(f"'{display}' is password protected.")

        page_count = len(doc)
        if page_count == 0:
            raise PDFValidationError(f"'{display}' has no pages.")

        if page_count > MAX_PAGES:
            raise PDFValidationError(
                f"'{display}' has {page_count} pages, over the {MAX_PAGES} page limit."
            )
    finally:
        doc.close()

    return page_count


def safe_display_name(original_name: str) -> str:
    """
    A version of the uploaded name that is safe to show and to use in citations.

    Strips any directory part and anything unusual, so a crafted filename cannot
    smuggle path separators or control characters into the UI.
    """
    base = Path(str(original_name)).name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(". ")
    cleaned = cleaned[:120]
    return cleaned or "document.pdf"


def safe_storage_path(upload_dir: Path) -> Path:
    """
    A random, collision-free path inside `upload_dir` to store an upload at.

    The uploaded name is never used on disk, so a hostile name cannot traverse
    out of the directory or overwrite a file that is already there.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)

    while True:
        candidate = upload_dir / f"{secrets.token_hex(16)}.pdf"
        if not candidate.exists():
            return candidate
