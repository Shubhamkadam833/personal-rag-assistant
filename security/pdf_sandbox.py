"""
pdf_sandbox.py — runs PDF text extraction in a separate process with a
hard time limit.

WHY:
  rag/pdf_loader.load_pdf() runs PyMuPDF directly in the Streamlit process. A
  PDF engineered to be pathologically slow to parse (or one that triggers a
  parser crash) would hang or kill the whole app, for every user, from one
  upload. Running extraction in a child process gets two things a plain
  function call cannot:

    1. A wall-clock timeout that actually works — Python cannot forcibly
       interrupt a C extension (PyMuPDF) running in-process, but a parent
       process CAN terminate a child process that has overrun its budget.
    2. Crash isolation — if the parser segfaults, the child process dies;
       the Streamlit server does not.

  This is process isolation with a timeout, not a security sandbox (no
  seccomp/container/filesystem restriction). It is the practical option
  available from the standard library, which is what this task asked for.

WHY `spawn`, EXPLICITLY:
  multiprocessing's default start method differs by platform (and `fork` is
  not available on Windows at all). Requesting `spawn` explicitly makes
  behavior the same on every platform this app might run on, rather than
  depending on an OS default.
"""

import multiprocessing
from pathlib import Path

from rag.pdf_loader import load_pdf
from security.pdf_validator import PDFValidationError

DEFAULT_TIMEOUT_SECONDS = 30

_ctx = multiprocessing.get_context("spawn")


def _worker(pdf_path: str, doc_name: str, result_queue: multiprocessing.Queue) -> None:
    """Runs in the child process. Must only communicate back via the queue."""
    try:
        pages = load_pdf(pdf_path, doc_name=doc_name)
        result_queue.put(("ok", pages))
    except Exception as e:
        # Exception objects from other modules may not unpickle cleanly in
        # the parent; send the message as a plain string instead.
        result_queue.put(("error", str(e)))


def extract_pages_with_timeout(
    pdf_path: str | Path,
    doc_name: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict]:
    """
    Extract text from a PDF already written to disk, in an isolated process,
    aborting if it takes longer than `timeout_seconds`.

    Raises PDFValidationError on timeout or on any extraction failure, with a
    message safe to show the user.
    """
    result_queue = _ctx.Queue()
    process = _ctx.Process(
        target=_worker,
        args=(str(pdf_path), doc_name, result_queue),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        raise PDFValidationError(
            f"'{doc_name}' took too long to process (over {timeout_seconds}s) and was stopped."
        )

    if result_queue.empty():
        raise PDFValidationError(f"'{doc_name}' could not be processed.")

    status, payload = result_queue.get()
    if status == "error":
        raise PDFValidationError(payload)

    return payload
