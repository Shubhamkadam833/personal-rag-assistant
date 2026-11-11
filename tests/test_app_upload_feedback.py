"""
App-level tests (Streamlit AppTest) for upload-area feedback:
- picking the same file twice must not crash the page (duplicate widget keys)
- Delete and Rebuild confirmations must still be visible after the page redraws

AppTest can't drive a real file upload, so st.file_uploader is replaced with
a stand-in that returns in-memory files. Nothing is written to data/uploads:
these tests never click "Add to index".
"""
import sys
from pathlib import Path

import fitz
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import document_registry, vector_store

APP = str(Path(__file__).parent.parent / "app.py")


class FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.name, self.size, self._data = name, len(data), data

    def getvalue(self) -> bytes:
        return self._data


def _pdf() -> bytes:
    doc = fitz.open(); doc.new_page().insert_text((50, 100), "hello")
    data = doc.tobytes(); doc.close()
    return data


@pytest.fixture(autouse=True)
def temp_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "EMBEDDINGS_FILE", tmp_path / "embeddings.npy")
    monkeypatch.setattr(vector_store, "METADATA_FILE", tmp_path / "metadata.json")
    monkeypatch.setattr(document_registry, "STORE_DIR", tmp_path)
    monkeypatch.setattr(document_registry, "REGISTRY_FILE", tmp_path / "documents.json")


def _seed_document(doc_id: str = "doc1") -> None:
    vector_store.add_chunks([{"doc_name": "report.pdf", "page": 1, "text": "x", "doc_id": doc_id}],
                            [np.ones(384).tolist()])
    # The stored file doesn't exist, so deleting it is a no-op on the real uploads folder.
    document_registry.add(doc_id, "report.pdf", "public", "no-such-test-file-0123.pdf")


def _success_after_click(at: AppTest, button) -> list[str]:
    """
    Messages visible once the click's st.rerun() has been drawn. AppTest
    sometimes applies that rerun itself and sometimes not, so redraw once more
    only if needed. The message must show once, so redrawing after it has
    already been shown would (correctly) clear it.
    """
    button.click().run()
    shown = [s.value for s in at.success]
    if not shown:
        at.run()
        shown = [s.value for s in at.success]
    return shown


def test_same_file_picked_twice_does_not_crash(monkeypatch):
    data = _pdf()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [FakeUpload("report.pdf", data), FakeUpload("report.pdf", data)])
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    assert any("selected twice" in i.value for i in at.info)


def test_delete_confirmation_survives_the_redraw():
    _seed_document()
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("Removed report.pdf" in s for s in _success_after_click(at, at.button(key="del_doc1")))


def test_rebuild_confirmation_survives_the_redraw():
    # Guards the new behavior only: for this button AppTest shows the frame
    # drawn before st.rerun(), so it can't reproduce the old vanishing
    # message. That was checked in a real browser (old: never visible; new:
    # visible from 0.4 s), see the PR description.
    _seed_document()
    at = AppTest.from_file(APP, default_timeout=120).run()
    rebuild = next(b for b in at.button if b.label == "Rebuild index")
    assert any("Index cleared" in s for s in _success_after_click(at, rebuild))
