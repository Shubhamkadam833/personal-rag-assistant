"""Tests for conversation history trimming and markdown escaping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.answer_generator import trim_history
from app import render_answer


def test_empty_history():
    assert trim_history(None) == []
    assert trim_history([]) == []


def test_keeps_only_role_and_content():
    history = [{"role": "assistant", "content": "Revenue grew 16.5%.", "sources": [{"doc_name": "x.pdf", "page": 2}]}]
    assert trim_history(history) == [{"role": "assistant", "content": "Revenue grew 16.5%."}]


def test_caps_to_max_turns():
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    trimmed = trim_history(history, max_turns=6)
    assert len(trimmed) == 6
    assert trimmed[-1]["content"] == "q19", "Should keep the most recent turns"


def test_skips_blank_and_unknown_roles():
    history = [
        {"role": "user", "content": "real question"},
        {"role": "system", "content": "not a conversation turn"},
        {"role": "assistant", "content": ""},
    ]
    assert trim_history(history) == [{"role": "user", "content": "real question"}]


def test_escapes_dollar_amounts():
    # The reported bug: text between two dollar amounts rendered as LaTeX math.
    raw = "free cash flow of $71m and adjusted EBITDA of $313m"
    assert render_answer(raw) == r"free cash flow of \$71m and adjusted EBITDA of \$313m"


def test_escaping_leaves_other_text_untouched():
    assert render_answer("no currency here") == "no currency here"
