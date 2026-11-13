"""
text_chunker.py — splits long page text into smaller overlapping chunks.

WHY WE CHUNK:
  AI models have a limit on how much text they can process at once.
  Also, smaller focused chunks give better search results than whole pages.

HOW OVERLAP WORKS:
  If chunk_size=150 and overlap=30, chunk 2 starts 120 words into chunk 1.
  This prevents an answer from being split across two chunks.

CHOOSING chunk_size:
  Sizes are counted in WORDS, not characters. Chunks must be small enough that
  retrieving the top few gives a focused context — if a document produces fewer
  chunks than top_k, every search returns the whole document and the semantic
  search does nothing.

THREE STRATEGIES (chunk_document; details in docs/chunking.md):
  fixed      Word windows within each page. Simple; can cut a sentence, a
             table row or a heading away from its text.
  paragraph  Packs whole paragraphs up to chunk_size words, across pages if
             needed. A long paragraph is split at line breaks, never mid-line,
             so a table row stays whole. A chunk never ends on a heading.
  section    Like paragraph, but never crosses a heading; the heading is
             repeated at the top of every chunk of its section.

WHAT IT RETURNS:
  A list of chunk dicts: doc_name, page (the first page), pages (every page
  the chunk spans), section (nearest heading, or ""), position, and text.
"""

import statistics

STRATEGIES = ("fixed", "paragraph", "section")


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict]:
    """Split page text into overlapping word-level chunks."""
    # Each chunk starts chunk_size - overlap words after the last one; at zero
    # or below the loop never ends.
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")
    chunks = []

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        step = chunk_size - overlap
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "doc_name": page["doc_name"],
                "page": page["page"],
                "text": chunk_text,
            })

            if end == len(words):
                break
            start += step

    return chunks


def chunk_document(
    pages: list[dict],
    strategy: str = "fixed",
    chunk_size: int = 150,
    overlap: int = 30,
) -> list[dict]:
    """Split one document's pages into chunks with the chosen strategy."""
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"need 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")
    if strategy == "fixed":
        chunks = [dict(c, pages=[c["page"]], section="") for c in chunk_pages(pages, chunk_size, overlap)]
    elif strategy in ("paragraph", "section"):
        units = _paragraphs(pages)
        groups = _split_at_headings(units) if strategy == "section" else [units]
        chunks = []
        for group in groups:
            heading = group[0]["text"] if strategy == "section" and group[0]["heading"] else None
            chunks += _pack(group, chunk_size, overlap, heading)
    else:
        raise ValueError(f"unknown chunk strategy {strategy!r}; use one of {STRATEGIES}")
    for i, chunk in enumerate(chunks):
        chunk["position"] = i
    return chunks


def _paragraphs(pages: list[dict]) -> list[dict]:
    """One unit per text block, marked as heading or not, with its nearest heading as section."""
    raw = []
    for page in pages:
        blocks = page.get("blocks") or [{"lines": page["text"].splitlines(), "size": None, "bold": False}]
        for b in blocks:
            lines = [ln.strip() for ln in b["lines"] if ln.strip()]
            if lines:
                raw.append((page, lines, b.get("size"), b.get("bold", False)))

    # Normal text size = the size most words are written in.
    sizes = [size for _, lines, size, _ in raw if size for _ in " ".join(lines).split()]
    body = statistics.median(sizes) if sizes else None

    units, section = [], ""
    for page, lines, size, bold in raw:
        text = "\n".join(lines)
        words = len(text.split())
        bigger = bool(body and size and size >= body * 1.15)
        heading = words <= 20 and (bigger or (bold and words <= 12))
        if heading:
            section = text.replace("\n", " ")[:120]
        units.append({"doc_name": page["doc_name"], "page": page["page"], "lines": lines,
                      "text": text, "words": words, "heading": heading, "section": section})
    return units


def _split_at_headings(units: list[dict]) -> list[list[dict]]:
    groups = []
    for unit in units:
        if unit["heading"] or not groups:
            groups.append([])
        groups[-1].append(unit)
    return groups


def _pieces(units: list[dict], chunk_size: int):
    """Units no longer than chunk_size: long ones split at line breaks, and a single over-long line by words."""
    for unit in units:
        if unit["words"] <= chunk_size:
            yield unit
            continue
        lines = []
        for line in unit["lines"]:
            words = line.split()
            lines += [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        piece = []
        for line in lines:
            if piece and len(" ".join(piece + [line]).split()) > chunk_size:
                yield _piece(unit, piece)
                piece = []
            piece.append(line)
        if piece:
            yield _piece(unit, piece)


def _piece(unit: dict, lines: list[str]) -> dict:
    text = "\n".join(lines)
    return dict(unit, lines=lines, text=text, words=len(text.split()), heading=False)


def _pack(units: list[dict], chunk_size: int, overlap: int, heading: str | None) -> list[dict]:
    """Greedily fill chunks with whole units, repeating up to `overlap` words of the previous one."""
    chunks, cur, carried = [], [], 0  # `carried` leading units of cur are overlap, not new text

    def count(us):
        return sum(u["words"] for u in us)

    def flush():
        nonlocal cur, carried
        held = []
        while len(cur) > carried and cur[-1]["heading"]:
            held.insert(0, cur.pop())  # a heading opens the next chunk, never ends one
        tail = []
        if len(cur) > carried:
            chunks.append(_make(cur, heading))
            for u in reversed(cur):
                if count(tail) + u["words"] > overlap:
                    break
                tail.insert(0, u)
        cur, carried = tail + held, len(tail)

    for unit in _pieces(units, chunk_size):
        if len(cur) > carried and count(cur) + unit["words"] > chunk_size:
            flush()
            if count(cur) + unit["words"] > chunk_size:  # overlap leaves no room: drop it
                cur, carried = cur[carried:], 0
        cur.append(unit)
    if len(cur) > carried:
        chunks.append(_make(cur, heading))
    return chunks


def _make(units: list[dict], heading: str | None) -> dict:
    text = "\n".join(u["text"] for u in units)
    if heading and units[0]["text"] != heading:
        text = f"{heading}\n{text}"  # keep the section heading with every part of its section
    return {
        "doc_name": units[0]["doc_name"],
        "page": units[0]["page"],
        "pages": sorted({u["page"] for u in units}),
        "section": units[0]["section"],
        "text": text,
    }
