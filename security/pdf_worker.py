"""
pdf_worker.py — the child-process half of pdf_sandbox.py.

Run as `python -m security.pdf_worker <pdf_path> <doc_name>`. Prints one JSON
line to stdout: {"ok": true, "pages": [...]} or {"ok": false, "error": "..."}.
Kept as its own tiny module so the child imports only what it needs, never
app.py.
"""

import json
import sys

from rag.pdf_loader import load_pdf


def main() -> None:
    pdf_path, doc_name = sys.argv[1], sys.argv[2]
    try:
        result = {"ok": True, "pages": load_pdf(pdf_path, doc_name=doc_name)}
    except ValueError as e:
        # load_pdf's messages are safe to show; the underlying cause (which may
        # name paths) goes to stderr, which pdf_sandbox logs to the terminal.
        cause = e.__cause__ or e
        print(f"{type(cause).__name__}: {cause}", file=sys.stderr)
        result = {"ok": False, "error": str(e)}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
