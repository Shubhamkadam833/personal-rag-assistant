"""
text_chunker.py — splits long page text into smaller overlapping chunks.

WHY WE CHUNK:
  AI models have a limit on how much text they can process at once.
  Also, smaller focused chunks give better search results than whole pages.

HOW OVERLAP WORKS:
  If chunk_size=500 and overlap=50, chunk 2 starts 450 words into chunk 1.
  This prevents an answer from being split across two chunks.

WHAT IT RETURNS:
  A list of chunk dicts — each chunk carries the source doc name and page number.
"""


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split page text into overlapping word-level chunks."""
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
