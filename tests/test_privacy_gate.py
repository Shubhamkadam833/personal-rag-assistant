"""Tests that Personal content can't reach Claude through history or query rewriting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query_rewriter import build_transcript
from security.privacy_gate import needs_approval, privacy_levels_of

PUBLIC = {"text": "x", "privacy": "public"}
PERSONAL = {"text": "y", "privacy": "personal"}


def test_personal_chunk_needs_approval():
    assert needs_approval([PUBLIC, PERSONAL], []) is True


def test_public_only_no_history_needs_no_approval():
    assert needs_approval([PUBLIC], []) is False


def test_public_chunks_with_personal_history_still_needs_approval():
    # The leak this fixes: a Public-only follow-up resending a Personal-derived answer.
    history = [
        {"role": "user", "content": "What is my salary?"},
        {"role": "assistant", "content": "Your salary is 90k.", "used_privacy": ["personal"]},
    ]
    assert needs_approval([PUBLIC], history) is True


def test_public_history_needs_no_approval():
    history = [{"role": "assistant", "content": "Revenue grew.", "used_privacy": ["public"]}]
    assert needs_approval([PUBLIC], history) is False


def test_untagged_legacy_history_needs_no_approval():
    # Messages from before tagging existed carry no used_privacy key.
    assert needs_approval([PUBLIC], [{"role": "assistant", "content": "old"}]) is False


def test_privacy_levels_of_deduplicates():
    assert privacy_levels_of([PUBLIC, PERSONAL, PUBLIC]) == ["personal", "public"]


def test_rewriter_transcript_excludes_assistant_answers():
    history = [
        {"role": "user", "content": "What was revenue in 2025?"},
        {"role": "assistant", "content": "SECRET personal fact from a document."},
    ]
    transcript = build_transcript(history)
    assert "What was revenue in 2025?" in transcript
    assert "SECRET" not in transcript


def test_rewriter_transcript_keeps_only_recent_turns():
    # Same window as the answer call, so the prompt doesn't grow all session.
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    assert build_transcript(history).splitlines() == [f"User: q{i}" for i in range(14, 20)]


def test_rewriter_transcript_empty_without_user_turns():
    assert build_transcript([{"role": "assistant", "content": "hi"}]) == ""
    assert build_transcript(None) == ""
