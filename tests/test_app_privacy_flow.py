"""
App-level tests (Streamlit AppTest) for the Personal-data approval flow: the
prompt must appear before any Personal content reaches the answer model,
approving must send once and tag the answer, declining must send nothing, and
a follow-up that resends a Personal answer as history must ask again.

A fake Claude client counts calls: no network, no API key, no cost. A temp
index with fixed vectors stands in for real documents.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import answer_generator, document_registry, retriever, vector_store

APP = str(Path(__file__).parent.parent / "app.py")
VEC = np.ones(384, dtype=np.float32).tolist()
calls: list[str] = []


class FakeClaude:
    def __init__(self, *args, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        calls.append(kwargs["model"])
        text = "Answer from the excerpts.\nUSED_EXCERPTS: 1"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


def answer_calls() -> int:
    """Calls to the answer model only (the follow-up rewriter is a separate, allowed call)."""
    return calls.count(answer_generator.MODEL)


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    calls.clear()
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    monkeypatch.setattr(retriever, "embed_query", lambda q: VEC)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClaude)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


def _add_doc(doc_id: str, privacy: str) -> None:
    vector_store.add_chunks([{"doc_name": f"{doc_id}.pdf", "page": 1, "text": f"{doc_id} text", "doc_id": doc_id}], [VEC])
    document_registry.add(doc_id, f"{doc_id}.pdf", privacy, f"{doc_id}.pdf")


def _ask(at: AppTest, question: str) -> None:
    at.chat_input[0].set_value(question).run()
    # When approval is needed the app calls st.rerun(). AppTest doesn't always
    # apply it; one more redraw is harmless (the pending approval persists).
    at.run()


def _prompt_showing(at: AppTest) -> bool:
    return any("Personal" in w.value for w in at.warning)


def _new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def test_personal_content_waits_for_approval():
    _add_doc("payslip", "personal")
    at = _new_app(); at.run()
    _ask(at, "What is my salary?")
    assert _prompt_showing(at)
    assert answer_calls() == 0, "Personal excerpts must not be sent before approval"


def test_approving_sends_once_and_tags_the_answer_personal():
    _add_doc("payslip", "personal")
    at = _new_app(); at.run()
    _ask(at, "What is my salary?")
    at.button(key="approve_personal").click().run()
    assert answer_calls() == 1
    assert "personal" in at.session_state["messages"][-1]["used_privacy"]


def test_declining_sends_nothing():
    _add_doc("payslip", "personal")
    at = _new_app(); at.run()
    _ask(at, "What is my salary?")
    at.button(key="cancel_personal").click().run()
    assert answer_calls() == 0
    assert "won't send" in at.session_state["messages"][-1]["content"]


def test_follow_up_resending_a_personal_answer_asks_again():
    _add_doc("report", "public")
    at = _new_app()
    at.session_state["messages"] = [
        {"role": "user", "content": "What is my salary?"},
        {"role": "assistant", "content": "Your salary is 90k.", "sources": [], "used_privacy": ["personal"]},
    ]
    at.run()
    _ask(at, "And what was revenue?")
    assert _prompt_showing(at)
    assert answer_calls() == 0


def test_public_only_question_is_answered_without_a_prompt():
    _add_doc("report", "public")
    at = _new_app(); at.run()
    at.chat_input[0].set_value("What was revenue?").run()
    assert not _prompt_showing(at)
    assert answer_calls() == 1
