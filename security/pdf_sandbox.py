"""
pdf_sandbox.py — runs PDF text extraction in a separate process with a
hard time limit.

WHY:
  rag/pdf_loader.load_pdf() runs PyMuPDF directly in the Streamlit process. A
  PDF engineered to be pathologically slow to parse (or one that triggers a
  parser crash) would hang or kill the whole app from one upload. Running
  extraction in a child process gets two things a plain function call cannot:

    1. A wall-clock timeout that actually works — Python cannot interrupt a
       C extension (PyMuPDF) running in-process, but it can kill a child
       process that overruns its budget.
    2. Crash isolation — if the parser segfaults, the child dies; the
       Streamlit server does not.

  This is process isolation with a timeout, not a security sandbox (no
  seccomp/container/filesystem restriction).

WHY A PLAIN SUBPROCESS, NOT multiprocessing:
  multiprocessing on Windows starts the child by re-importing the parent's
  main script first. Under Streamlit that script is app.py, so every upload
  booted a second copy of the whole app (embedding model included) and timed
  out before extraction even began. Launching `python -m security.pdf_worker`
  starts a clean interpreter that imports only the worker.
"""

import json
import subprocess  # nosec B404
import sys
from pathlib import Path

from security.pdf_validator import PDFValidationError

DEFAULT_TIMEOUT_SECONDS = 30
PROJECT_ROOT = Path(__file__).parent.parent


def extract_pages_with_timeout(
    pdf_path: str | Path,
    doc_name: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict]:
    """
    Extract text from a PDF already written to disk, in an isolated process,
    aborting if it takes longer than `timeout_seconds`.

    Raises PDFValidationError on timeout or on any extraction failure, with a
    message safe to show the user.
    """
    try:
        # Fixed interpreter and module; the path is one we generated ourselves.
        completed = subprocess.run(  # nosec B603
            [sys.executable, "-m", "security.pdf_worker", str(pdf_path), doc_name],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=PROJECT_ROOT,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child before raising.
        raise PDFValidationError(
            f"'{doc_name}' took too long to process (over {timeout_seconds}s) and was stopped."
        )

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        # Crash or unexpected output. stderr may hold paths, so don't show it.
        raise PDFValidationError(f"'{doc_name}' could not be processed.")

    if not result.get("ok"):
        raise PDFValidationError(result.get("error") or f"'{doc_name}' could not be processed.")
    return result["pages"]
