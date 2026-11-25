"""
App-level test (Streamlit AppTest) for reranking: the answer shows which
search method found the evidence, how ranks moved, and a clear notice when
reranking fell back; the toggle turns it off; Restricted documents never
appear. A fake Claude client and a fake reranker model: no network, no cost.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import bm25, document_registry, reranker, retriever, vector_store

APP = str(Path(__file__).parent.parent / "app.py")
VEC = np.ones(384, dtype=np.float32).tolist()
sent_prompts: list[str] = []


class FakeClaude:
    def __init__(self, *args, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        sent_prompts.append(str(kwargs["messages"]))
        # Cite the excerpt that holds the figure, as the real citations API would.
        docs = [b for b in kwargs["messages"][-1]["content"] if b["type"] == "document"]
        index = next(i for i, d in enumerate(docs) if "61.9" in d["source"]["data"])
        cite = {"document_index": index, "cited_text": docs[index]["source"]["data"]}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Operating income was 61.9.", citations=[cite])],
                               stop_reason="end_turn")


class FakeReranker:
    """Prefers the chunk that mentions FY2024."""

    def predict(self, pairs, show_progress_bar=False):
        return np.array([5.0 if "FY2024" in t else 0.0 for _, t in pairs], dtype=np.float32)


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    sent_prompts.clear()
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: VEC)
    monkeypatch.setattr(bm25, "_cache", None)
    monkeypatch.setattr(reranker, "_model", FakeReranker())
    monkeypatch.setattr(reranker, "_load_error", None)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClaude)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    # Search ranks FY2023 first (closer vector, more keyword hits); the reranker should flip it.
    far = np.array(VEC); far[:300] = 0.0
    texts = ["Operating income FY2023 was 47.8; operating income rose", "Operating income FY2024 was 61.9"]
    vector_store.add_chunks([{"doc_name": "report.pdf", "page": i + 1, "text": t, "doc_id": "rep"}
                             for i, t in enumerate(texts)], [VEC, far.tolist()])
    document_registry.add("rep", "report.pdf", "public", "report.pdf")
    vector_store.add_chunks([{"doc_name": "vault.pdf", "page": 1, "text": "Operating income secret", "doc_id": "vault"}], [VEC])
    document_registry.add("vault", "vault.pdf", "restricted", "vault.pdf")


def _evidence_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)


def test_reranked_answer_shows_method_and_rank_changes():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("What was operating income?").run()
    assert not at.exception
    shown = _evidence_text(at)
    assert "hybrid search + reranking" in shown
    assert "#1** (search rank #2) · report.pdf — page 2 · relevance score 5.00" in shown
    assert "not a chance of being correct" in shown
    # The sidebar lists every document by name; the evidence and the prompt must not include Restricted ones.
    assert {r["doc_name"] for r in at.session_state["messages"][-1]["evidence"]["rows"]} == {"report.pdf"}
    assert sent_prompts and "secret" not in " ".join(sent_prompts)


def test_fallback_is_announced_and_keeps_search_order(monkeypatch):
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_model", lambda: (_ for _ in ()).throw(OSError("offline")))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("What was operating income?").run()
    assert not at.exception
    assert any("Reranking unavailable (model unavailable)" in c.value for c in at.caption)
    assert at.session_state["messages"][-1]["content"].startswith("Operating income"), "still answered"


def test_turning_reranking_off_in_the_sidebar():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="rerank").set_value(False).run()
    at.selectbox(key="search_mode").set_value("semantic").run()
    at.chat_input[0].set_value("What was operating income?").run()
    evidence = at.session_state["messages"][-1]["evidence"]
    assert evidence["info"]["mode"] == "semantic" and not evidence["info"]["reranked"]
    assert all(r["score"] is None and r["final"] == r["original"] for r in evidence["rows"])
    assert "+ reranking" not in _evidence_text(at)
