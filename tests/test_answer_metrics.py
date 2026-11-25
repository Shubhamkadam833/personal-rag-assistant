"""Scoring rules of the Phase 5 answer benchmark. Pure functions: no API calls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.answer_metrics import (has_fact, is_abstention, is_correct, page_references, parse_judge, rate,
                                       split_statements)
from rag.answer_generator import NO_ANSWER, UNVERIFIED

FIGURE = {"expect": "answer", "facts": [["412.6"]]}
CONFLICT = {"expect": "conflict", "facts": [["41.8"], ["44.3"]]}
MISSING = {"expect": "abstain", "facts": []}


def test_both_refusal_messages_count_as_abstaining():
    assert is_abstention(NO_ANSWER) and is_abstention(UNVERIFIED)
    assert is_abstention("I couldn’t find this information in the uploaded documents.")  # curly apostrophe
    assert not is_abstention("Revenue was 412.6.")


def test_numbers_match_by_value_and_words_case_insensitively():
    assert has_fact("It had 4120 staff.", ["4,120"])
    assert has_fact("Margin was 16.0%.", ["16"]) and has_fact("Margin was 16%.", ["16.0"])
    assert has_fact("lumen grid led", ["Lumen"])
    assert not has_fact("Revenue was 412.7.", ["412.6"])


def test_correctness_rules_per_question_type():
    assert is_correct(FIGURE, "Revenue was $412.6 million. [1]")
    assert not is_correct(FIGURE, NO_ANSWER)
    assert is_correct(CONFLICT, "Restated to 41.8, from 44.3 reported earlier.")
    assert not is_correct(CONFLICT, "Net income was 41.8.")
    assert is_correct(MISSING, NO_ANSWER) and not is_correct(MISSING, "The ticker is NWR.")


def test_a_forbidden_phrase_makes_an_answer_wrong():
    q = {"expect": "answer", "facts": [["1,284.0"]], "forbidden": ["5 billion"]}
    assert is_correct(q, "Revenue was $1,284.0 million.")
    assert not is_correct(q, "Revenue was $1,284.0 million, or $5 billion per a note.")


def test_judge_lines_are_parsed_and_junk_is_left_out():
    text = "Here you go:\n1 | claim | yes | 1, 3 | n/a\n2 | filler | yes | - | n/a\n3 | CLAIM | No | 2 | no\n9 | claim | yes | 1 | yes"
    verdicts = parse_judge(text, 4)
    assert verdicts[0] == {"kind": "claim", "supported": True, "excerpts": [1, 3], "quote": "n/a"}
    assert verdicts[1]["kind"] == "filler"
    assert verdicts[2] == {"kind": "claim", "supported": False, "excerpts": [2], "quote": "no"}
    assert verdicts[3] is None, "a statement the judge skipped is not guessed"


def test_old_style_answers_are_split_without_their_source_lines():
    answer = "Revenue was 412.6. Margin was 15%.\n\nSource: report.pdf, page 2"
    assert split_statements(answer) == ["Revenue was 412.6.", "Margin was 15%."]
    assert page_references(answer) == [2]


def test_rate_is_none_when_there_is_nothing_to_count():
    assert rate(0, 0) is None and rate(1, 4) == 0.25
