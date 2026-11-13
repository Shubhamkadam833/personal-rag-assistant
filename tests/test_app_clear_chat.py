"""
App-level test (Streamlit's AppTest): clearing the chat must also discard a
pending Personal-data approval. Otherwise approving it afterwards would send
the old question plus the history the user just cleared.

No API call is made: the approval is never approved in this test.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, vector_store


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    # Never depend on (or touch) the real data folder.
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")


def test_clear_chat_discards_pending_approval():
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["messages"] = [{"role": "user", "content": "What is my salary?"}]
    at.session_state["pending_personal"] = {
        "question": "What is my salary?",
        "chunks": [{"text": "Salary 90k", "doc_name": "payslip.pdf", "page": 1, "privacy": "personal"}],
        "history": [{"role": "assistant", "content": "earlier private answer"}],
    }
    at.run()
    assert any("Personal" in w.value for w in at.warning), "approval prompt should be showing"

    at.button(key="clear_chat").click().run()
    # The Clear chat handler sits at the bottom of app.py, so the click's run
    # drew the prompt before clearing it; its st.rerun() then redraws the page.
    # AppTest didn't apply that rerun here, so redraw like the browser does.
    at.run()

    assert at.session_state["pending_personal"] is None
    assert at.session_state["messages"] == []
    assert not any("Personal" in w.value for w in at.warning), "approval prompt should be gone"
