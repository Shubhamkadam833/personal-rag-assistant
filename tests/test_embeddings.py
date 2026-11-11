"""Tests for rag/embeddings.py model loading — disk first, download only if missing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import embeddings


class _FakeModel:
    """Stands in for SentenceTransformer; records how it was constructed."""
    calls: list[dict] = []
    cached = True

    def __init__(self, name, local_files_only=False):
        _FakeModel.calls.append({"name": name, "local_files_only": local_files_only})
        if local_files_only and not _FakeModel.cached:
            raise OSError("not in local cache")


def _use_fake(monkeypatch, cached: bool):
    _FakeModel.calls = []
    _FakeModel.cached = cached
    monkeypatch.setattr(embeddings, "SentenceTransformer", _FakeModel)


def test_cached_model_loads_without_network(monkeypatch):
    _use_fake(monkeypatch, cached=True)
    embeddings._load_model()
    assert _FakeModel.calls == [{"name": embeddings.MODEL_NAME, "local_files_only": True}]


def test_missing_model_falls_back_to_one_download(monkeypatch):
    _use_fake(monkeypatch, cached=False)
    embeddings._load_model()
    assert [c["local_files_only"] for c in _FakeModel.calls] == [True, False]
