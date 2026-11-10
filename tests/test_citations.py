"""
Tests for rag/answer_generator.extract_used_excerpts — making sure a shown
source genuinely reflects what the model says it used, not just what was
retrieved. Pure function: no API key or network access needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.answer_generator import extract_used_excerpts


def test_parses_single_used_excerpt():
    raw = "Revenue grew 16.5%.\nUSED_EXCERPTS: 1"
    text, used = extract_used_excerpts(raw, excerpt_count=3)
    assert text == "Revenue grew 16.5%."
    assert used == [1]


def test_parses_multiple_used_excerpts():
    raw = "Combining two facts.\nUSED_EXCERPTS: 1, 3"
    text, used = extract_used_excerpts(raw, excerpt_count=3)
    assert used == [1, 3]


def test_none_means_no_sources():
    raw = "I couldn't find this information in the uploaded documents.\nUSED_EXCERPTS: none"
    text, used = extract_used_excerpts(raw, excerpt_count=5)
    assert used == []


def test_missing_marker_yields_no_sources():
    # The model must not get the benefit of the doubt — a missing marker
    # means we cannot show sources as "used", so we show none.
    raw = "Revenue grew 16.5% last year."
    text, used = extract_used_excerpts(raw, excerpt_count=3)
    assert used == []
    assert text == raw


def test_out_of_range_index_is_dropped():
    # Guards against the model citing an excerpt number that was never given.
    raw = "Answer text.\nUSED_EXCERPTS: 1, 99"
    text, used = extract_used_excerpts(raw, excerpt_count=3)
    assert used == [1]


def test_non_numeric_entries_are_dropped():
    raw = "Answer text.\nUSED_EXCERPTS: 1, two, 3"
    text, used = extract_used_excerpts(raw, excerpt_count=3)
    assert used == [1, 3]


def test_marker_is_stripped_from_displayed_text():
    raw = "Line one.\nLine two.\nUSED_EXCERPTS: 2"
    text, used = extract_used_excerpts(raw, excerpt_count=2)
    assert "USED_EXCERPTS" not in text
    assert text == "Line one.\nLine two."


def test_only_cites_what_is_claimed_not_everything_retrieved():
    # The core behavior this requirement is about: 5 excerpts given, model
    # only used 2 of them — sources must reflect 2, not all 5.
    raw = "Mixed answer.\nUSED_EXCERPTS: 2, 4"
    text, used = extract_used_excerpts(raw, excerpt_count=5)
    assert used == [2, 4]
    assert len(used) < 5
