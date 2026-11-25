"""
citation_check.py — checks that an answer's citations are real and back up what it says.

Runs in Python, on your computer, after (and just before) the Claude call.
It costs nothing and does not depend on the model following instructions.

TWO DIFFERENT QUESTIONS:
  1. Is the citation VALID?  Does it point to an excerpt we actually sent, and
     is its quote really inside that excerpt? (Document, chunk and page come
     from the index, never from the model's own words.)
  2. Is the claim SUPPORTED?  A valid citation can still be attached to the
     wrong sentence: a real page about FY2023 revenue cited for an FY2024
     figure. Here we check that every number in the sentence appears in the
     passage it quotes. In financial answers the numbers are where a wrong
     answer does the most damage.

BEFORE THE CALL (current_chunks):
  An approval prompt can sit open while you delete a document, or delete it
  and upload it again as Restricted. So right before sending, every excerpt is
  checked again against the live index and registry: same chunk ID, document
  ID, page and text, and the same privacy level as when it was retrieved.
  Anything that fails is dropped, never sent.

HONEST LIMITS:
  Number matching is pattern matching, not understanding. It catches wrong,
  swapped or invented figures. It does not catch a wrong word ("increased"
  for "decreased") or an invented claim with no number in it, and it flags
  a correct calculation (a growth figure that appears in no document).
"""

import re

from rag import document_registry, vector_store

# A text block with no citation and no figure, shorter than this, is treated
# as connecting text ("Here is what I found:"), not as a factual statement.
MIN_STATEMENT_WORDS = 6

_LIST_MARKER = re.compile(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+")
_NUMBER = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?")

REASONS = {
    "figure_not_in_quote": "contains a figure that isn't in the passage it cites (it may be a calculation)",
    "no_quote": "contains a figure that no quoted passage contains (it may be a calculation)",
    "invalid_citation": "cites a passage that doesn't match the excerpt it points to",
    "not_quoted": "has no quote of its own, but no unquoted figure either",
}


def numbers(text: str) -> set[float]:
    """Every number in `text`, as a value: '$1,284.0' -> 1284.0, 'FY2024' -> 2024. List bullets are ignored."""
    found = set()
    for raw in _NUMBER.findall(_LIST_MARKER.sub(" ", text)):
        try:
            found.add(float(raw.rstrip(",").replace(",", "")))
        except ValueError:  # a stray "," on its own
            continue
    return found


def current_chunks(chunks: list[dict]) -> list[dict]:
    """The excerpts that still exist unchanged in the index and are still allowed to be sent."""
    registry = document_registry.list_all()
    _, metadata = vector_store._load()
    live = {m["hash"]: m for m in metadata if "hash" in m}
    kept = []
    for chunk in chunks:
        entry = live.get(chunk.get("chunk_id"))
        record = registry.get(chunk.get("doc_id"))
        if (entry and record
                and record["privacy"] in ("public", "personal")
                and record["privacy"] == chunk.get("privacy", record["privacy"])
                and entry.get("doc_id") == chunk["doc_id"]
                and entry["page"] == chunk["page"]
                and entry["text"] == chunk["text"]):
            kept.append(chunk)
    return kept


def _get(obj, key: str):
    # shortcut: anthropic 0.40 returns citations as plain dicts, newer SDKs as
    # objects; drop the dict branch once the SDK is upgraded.
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def text_blocks(content: list) -> list[dict]:
    """The response's text blocks, each with its citations: [{text, citations: [{index, quote}]}]."""
    blocks = []
    for block in content or []:
        if _get(block, "type") != "text":
            continue  # thinking blocks and anything else
        cites = [{"index": _get(c, "document_index"), "quote": _get(c, "cited_text") or ""}
                 for c in _get(block, "citations") or []]
        blocks.append({"text": _get(block, "text") or "", "citations": cites})
    return blocks


def citation_valid(cite: dict, chunks: list[dict]) -> bool:
    """Points at an excerpt we sent, and its quote is really inside that excerpt."""
    index = cite["index"]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(chunks):
        return False
    quote = cite["quote"].strip()
    return bool(quote) and quote in chunks[index]["text"]


def check_claims(blocks: list[dict], chunks: list[dict]) -> list[dict]:
    """
    One entry per factual statement: {text, status, cited (excerpt indexes), quotes, missing (figures)}.
    status: supported | figure_not_in_quote | no_quote (a figure quoted nowhere in the answer) |
            invalid_citation | not_quoted (no quote, and no figure that isn't quoted elsewhere: informational)
    """
    # Summaries often repeat a figure that is quoted elsewhere in the same answer
    # ("Net income was 41.8 ... restated to 41.8 [1]"); such a figure is backed.
    quoted_anywhere = numbers(" ".join(c["quote"] for b in blocks for c in b["citations"] if citation_valid(c, chunks)))
    claims = []
    for block in blocks:
        text = block["text"].strip()
        if not re.search(r"[A-Za-z0-9]", text):
            continue
        valid = [c for c in block["citations"] if citation_valid(c, chunks)]
        figures = numbers(text)
        missing: set[float] = set()
        if valid:
            missing = figures - numbers(" ".join(c["quote"] for c in valid))
            status = "figure_not_in_quote" if missing else "supported"
        elif block["citations"]:
            status = "invalid_citation"
        elif figures - quoted_anywhere:
            missing = figures - quoted_anywhere
            status = "no_quote"
        elif figures or len(re.findall(r"[A-Za-z]+", text)) >= MIN_STATEMENT_WORDS:
            # Framing like "The two documents disagree", or a summary whose figures
            # are all quoted elsewhere. Kept so the evaluation judges it, but it
            # doesn't change the answer's status.
            status = "not_quoted"
        else:
            continue  # connecting text
        claims.append({
            "text": text,
            "status": status,
            "cited": sorted({c["index"] for c in valid}),
            "quotes": [c["quote"].strip() for c in valid],
            "missing": sorted(missing),
        })
    return claims


def problems(claims: list[dict]) -> list[dict]:
    """The statements that fail a check (not_quoted framing text is informational, not a failure)."""
    return [c for c in claims if c["status"] not in ("supported", "not_quoted")]


def overall_status(claims: list[dict]) -> str:
    """verified: nothing fails; partial: some statements fail; unverified: no statement is backed."""
    if not any(c["status"] == "supported" for c in claims):
        return "unverified"
    return "partial" if problems(claims) else "verified"
