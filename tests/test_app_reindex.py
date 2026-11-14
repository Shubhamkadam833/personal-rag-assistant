"""
App-level test (Streamlit AppTest): an index built with other settings (here:
before settings were recorded) blocks the page with a Re-index button instead
of mixing vectors, and a document whose stored file is missing is reported,
not silently dropped. Reads data/uploads only; writes only the temp index.
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


def test_outdated_index_offers_reindex_and_reports_missing_files(temp_index):
    vector_store.add_chunks([{"doc_name": "report.pdf", "page": 1, "text": "x", "doc_id": "doc1"}],
                            [np.ones(384).tolist()])
    document_registry.add("doc1", "report.pdf", "public", "no-such-stored-file-0123.pdf")
    (temp_index / "manifest.json").unlink()  # as if built before settings were recorded

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("settings changed" in w.value for w in at.warning)

    at.button(key="reindex").click().run()
    shown = [s.value for s in at.success]
    if not shown:
        at.run()  # AppTest sometimes doesn't apply the button's st.rerun() itself
        shown = [s.value for s in at.success]
    assert any("please re-upload: report.pdf" in s for s in shown)
    assert document_registry.get("doc1") is not None, "kept in the registry, not silently dropped"
    vector_store.check_compatible()  # the rebuilt index matches today's settings
