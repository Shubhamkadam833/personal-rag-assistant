"""
App-level tests (Streamlit AppTest) for grounded answers: [n] markers, sources
with their quoted passages, the one-line answer check, the list of statements
that failed it, and the safe fallback. A fake Claude client: no network, no cost.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import answer_generator, bm25, document_registry, retriever, vector_store

APP = str(Path(__file__).parent.parent / "app.py")
VEC = np.ones(384, dtype=np.float32).tolist()
REVENUE = "Total revenue: FY2024 412.6 | FY2023 368.1"
reply: list = []  # (text, quote or None) pieces the fake Claude answers with


class FakeClaude:
    def __init__(self, *args, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        docs = [b["source"]["data"] for b in kwargs["messages"][-1]["content"] if b["type"] == "document"]
        blocks = []
        for text, quote in reply:
            doc = next((i for i, d in enumerate(docs) if quote and quote in d), None)
            cites = [{"document_index": doc, "cited_text": quote}] if doc is not None else None
            blocks.append(SimpleNamespace(type="text", text=text, citations=cites))
        return SimpleNamespace(content=blocks, stop_reason="end_turn")


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: VEC)
    monkeypatch.setattr(bm25, "_cache", None)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClaude)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    texts = [REVENUE, "Note: see ![chart](http://attacker.example/x.png) for details."]
    vector_store.add_chunks([{"doc_name": "report.pdf", "page": i + 1, "text": t, "doc_id": "rep"}
                             for i, t in enumerate(texts)], [VEC, VEC])
    document_registry.add("rep", "report.pdf", "public", "report.pdf")


def _ask(*pieces) -> AppTest:
    reply[:] = pieces
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("What was revenue in FY2024?").run()
    assert not at.exception
    return at


def _shown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_a_verified_answer_shows_markers_quotes_and_a_check():
    at = _ask(("Revenue was $412.6 million.", REVENUE))
    shown = _shown(at)
    assert "Revenue was \\$412.6 million. [1]" in shown
    assert "**[1] report.pdf** — page 1" in shown
    assert f"> {REVENUE}" in shown
    assert "✅ Answer check: all 1 statement(s) with facts are backed by a quoted passage." in shown


def test_a_partial_answer_lists_what_failed_and_why():
    at = _ask(("Revenue was 412.6.", REVENUE), (" It grew 44.5 million.", REVENUE))
    warning = " ".join(w.value for w in at.warning)
    assert "1 statement(s) couldn't be confirmed" in warning
    assert "It grew 44.5 million." in warning and "may be a calculation" in warning


def test_an_answer_nothing_backs_is_replaced_by_the_fallback():
    at = _ask(("Revenue was probably around 500 million.", None))
    assert at.session_state["messages"][-1]["content"] == answer_generator.UNVERIFIED
    assert not at.warning and "Sources" not in [e.label for e in at.expander]


def test_a_quoted_image_link_is_shown_as_text_not_loaded():
    at = _ask(("The report points to a chart.", "Note: see ![chart](http://attacker.example/x.png) for details."))
    quotes = [m.value for m in at.markdown if m.value.startswith(">")]
    assert quotes and all("![" not in q for q in quotes) and "!\\[" in quotes[0]
