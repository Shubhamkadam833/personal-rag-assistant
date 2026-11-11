"""
App-level test (Streamlit AppTest): a damaged index must not crash the page.
It shows a clear message and a working Rebuild button instead.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, vector_store

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")
    return tmp_path


def test_damaged_index_shows_a_message_and_rebuild_recovers(temp_index):
    np.save(str(temp_index / "embeddings.npy"), np.zeros((1, 384), dtype=np.float32))
    (temp_index / "metadata.json").write_text('[{"hash": "abc", "doc_na')  # cut short mid-write

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, "a damaged index must not crash the page"
    assert any("damaged" in e.value for e in at.error)

    at.button(key="rebuild_damaged").click().run()
    at.run()  # the button's st.rerun(); AppTest doesn't follow it on its own
    assert not at.exception
    assert not any("damaged" in e.value for e in at.error)
    assert not (temp_index / "metadata.json").exists()
