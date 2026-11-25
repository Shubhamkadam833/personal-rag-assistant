"""
Grounded answers: sources must reflect what Claude actually quoted, every
citation and figure is checked in code, and nothing changed or Restricted is
sent. A fake client stands in for the API: no network, no real key.

Responses are shaped like the real API's citations output: text blocks, each
cited block carrying {document_index, cited_text}. (anthropic 0.40 hands
citations over as plain dicts, which is what these use.)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import answer_generator, document_registry, vector_store
from rag.citation_check import numbers

REVENUE = "Total revenue: FY2024 412.6 | FY2023 368.1"
MARGIN = "Operating margin: FY2024 15.0% | FY2023 13.0%"


def text(t, *cites):
    """A response text block; each cite is (excerpt index, quoted text)."""
    return SimpleNamespace(type="text", text=t, citations=[
        {"type": "char_location", "document_index": i, "cited_text": q} for i, q in cites] or None)


class FakeClaude:
    sent: list[dict] = []
    reply: list = []
    stop_reason = "end_turn"

    def __init__(self, *args, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        FakeClaude.sent.append(kwargs)
        return SimpleNamespace(content=FakeClaude.reply, stop_reason=FakeClaude.stop_reason)


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(answer_generator.anthropic, "Anthropic", FakeClaude)
    FakeClaude.sent, FakeClaude.reply, FakeClaude.stop_reason = [], [], "end_turn"


def answer(chunks, *blocks, stop_reason="end_turn"):
    FakeClaude.reply, FakeClaude.stop_reason = list(blocks), stop_reason
    return answer_generator.generate_answer("What was revenue?", chunks)


def sent_text() -> str:
    """Everything the fake API received for the last call, as one string."""
    content = FakeClaude.sent[-1]["messages"][-1]["content"]
    return " ".join(b["source"]["data"] if b["type"] == "document" else b["text"] for b in content)


# --- guarantees carried over from the old USED_EXCERPTS tests ----------------

def test_a_single_quoted_excerpt_is_the_only_source(index_doc):
    chunks = index_doc("nw", [REVENUE, MARGIN])
    result = answer(chunks, text("Revenue was $412.6 million in FY2024.", (0, REVENUE)))
    assert result["answer"] == "Revenue was $412.6 million in FY2024. [1]"
    assert [(s["n"], s["doc_name"], s["page"], s["quotes"]) for s in result["sources"]] == [(1, "nw.pdf", 1, [REVENUE])]
    assert result["check"]["status"] == "verified"


def test_several_quoted_excerpts_are_numbered_in_order(index_doc):
    chunks = index_doc("nw", [REVENUE, MARGIN])
    result = answer(chunks, text("Margin was 15.0%.", (1, MARGIN)), text(" Revenue was 412.6.", (0, REVENUE)))
    assert result["answer"] == "Margin was 15.0%. [1] Revenue was 412.6. [2]"
    assert [s["page"] for s in result["sources"]] == [2, 1]


def test_only_quoted_excerpts_become_sources_not_everything_sent(index_doc):
    chunks = index_doc("nw", [REVENUE, MARGIN, "filler one", "filler two", "filler three"])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), text(" Margin was 15.0%.", (1, MARGIN)))
    assert len(result["sources"]) == 2 < len(chunks)


def test_saying_the_documents_lack_it_gives_no_sources(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text(answer_generator.NO_ANSWER))
    assert result["answer"] == answer_generator.NO_ANSWER
    assert result["sources"] == [] and result["check"]["status"] == "abstained"


def test_an_answer_with_no_citations_is_replaced_not_shown(index_doc):
    # The model doesn't get the benefit of the doubt: nothing backed, nothing shown.
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue grew strongly to about 500 million last year."))
    assert result["answer"] == answer_generator.UNVERIFIED
    assert result["sources"] == [] and result["check"]["status"] == "unverified"


def test_a_citation_to_an_excerpt_never_sent_is_rejected(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), text(" Profit was 99.", (7, "Profit was 99")))
    assert [s["n"] for s in result["sources"]] == [1]
    assert result["check"]["valid_citations"] == 1 and result["check"]["citations"] == 2
    assert result["check"]["claims"][1]["status"] == "invalid_citation"


def test_a_quote_that_is_not_in_its_excerpt_is_rejected(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 999.", (0, "Total revenue: FY2024 999")))
    assert result["answer"] == answer_generator.UNVERIFIED


def test_source_details_come_from_the_index_not_the_model(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 412.6, according to other.pdf.", (0, REVENUE)))
    src = result["sources"][0]
    assert (src["doc_id"], src["doc_name"], src["page"], src["chunk_id"]) == ("nw", "nw.pdf", 1, chunks[0]["chunk_id"])


# --- supported vs merely valid ----------------------------------------------

def test_a_real_citation_with_the_wrong_figure_is_flagged(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), text(" It rose 44.5.", (0, REVENUE)))
    assert result["check"]["status"] == "partial"
    claim = result["check"]["claims"][1]
    assert claim["status"] == "figure_not_in_quote" and claim["missing"] == [44.5]
    assert "[1]" in result["answer"], "a partial answer is still shown, with its sources"


def test_an_unquoted_figure_is_flagged_but_framing_text_is_not(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("The two documents in front of me disagree on this point. "),
                    text("Revenue was 412.6.", (0, REVENUE)), text(" Growth was 12.1%."))
    statuses = [c["status"] for c in result["check"]["claims"]]
    assert statuses == ["not_quoted", "supported", "no_quote"]
    assert result["check"]["status"] == "partial"


def test_framing_text_alone_does_not_make_an_answer_partial(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("The documents give two different figures for this. "), text("Revenue was 412.6.", (0, REVENUE)))
    assert result["check"]["status"] == "verified"


def test_a_summary_repeating_a_quoted_figure_is_not_flagged(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("In short, revenue was 412.6 in FY2024.\n"), text("Revenue: 412.6.", (0, REVENUE)))
    assert [c["status"] for c in result["check"]["claims"]] == ["not_quoted", "supported"]
    assert result["check"]["status"] == "verified"


def test_the_same_quote_from_overlapping_chunks_is_one_source(index_doc):
    chunks = index_doc("nw", [f"Intro. {REVENUE}", f"{REVENUE} More."])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), text(" Again 412.6.", (1, REVENUE)))
    assert len(result["sources"]) == 1 and result["answer"].count("[1]") == 2


def test_numbers_are_compared_as_values():
    assert numbers("$1,284.0 million in FY2024, up 12.1%") == {1284.0, 2024.0, 12.1}
    assert numbers("1. Revenue was 4,120.\n- 2. item") == {4120.0, 2.0}
    assert numbers("no figures here") == set()


# --- request shape and other outcomes ---------------------------------------

def test_excerpts_are_sent_as_documents_with_citations_and_no_marker_rule(index_doc):
    chunks = index_doc("nw", [REVENUE, MARGIN])
    answer(chunks, text(answer_generator.NO_ANSWER))
    content = FakeClaude.sent[-1]["messages"][-1]["content"]
    docs = [b for b in content if b["type"] == "document"]
    assert [d["source"]["data"] for d in docs] == [REVENUE, MARGIN]
    assert all(d["citations"] == {"enabled": True} and d["title"] == "nw.pdf" for d in docs)
    assert content[-1] == {"type": "text", "text": "Question: What was revenue?"}
    assert "USED_EXCERPTS" not in FakeClaude.sent[-1]["system"]
    assert FakeClaude.sent[-1]["max_tokens"] == answer_generator.MAX_TOKENS


def test_a_refusal_gets_a_clear_message(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), stop_reason="refusal")
    assert result["answer"] == answer_generator.REFUSED and result["sources"] == []


def test_a_cut_off_answer_says_so(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)), stop_reason="max_tokens")
    assert result["answer"].endswith(answer_generator.CUT_OFF)


def test_an_api_key_in_a_quoted_passage_is_not_shown(index_doc):
    leaked = "The key is sk-ant-abcdefghijklmnop123"
    chunks = index_doc("nw", [leaked])
    result = answer(chunks, text("Here it is.", (0, leaked)))
    assert "sk-ant-" not in str(result)


# --- privacy: re-checked right before sending ---------------------------------

def test_a_document_deleted_while_waiting_for_approval_is_not_sent(index_doc):
    chunks = index_doc("nw", [REVENUE])
    vector_store.remove_document("nw")
    document_registry.remove("nw")
    result = answer(chunks, text("Revenue was 412.6.", (0, REVENUE)))
    assert FakeClaude.sent == [], "nothing may be sent"
    assert result["answer"] == answer_generator.NO_ANSWER and result["check"]["dropped_excerpts"] == 1


def test_a_document_reuploaded_as_restricted_while_waiting_is_not_sent(index_doc):
    personal = index_doc("pay", ["Salary: 90,000"], privacy="personal")
    public = index_doc("nw", [REVENUE])
    vector_store.remove_document("pay")
    document_registry.remove("pay")
    index_doc("pay", ["Salary: 90,000"], privacy="restricted")  # same file, same ID, now Restricted

    answer(personal + public, text("Revenue was 412.6.", (0, REVENUE)))
    assert "Salary" not in sent_text() and REVENUE in sent_text()


def test_a_restricted_chunk_handed_in_by_mistake_is_never_sent(index_doc):
    secret = index_doc("vault", ["Merger codename Falcon"], privacy="restricted")
    public = index_doc("nw", [REVENUE])
    answer([{**secret[0], "privacy": "public"}] + public, text("Revenue was 412.6.", (0, REVENUE)))
    assert "Falcon" not in sent_text()


def test_an_excerpt_whose_text_changed_is_not_sent(index_doc):
    chunks = index_doc("nw", [REVENUE])
    result = answer([{**chunks[0], "text": "Total revenue: FY2024 999.0"}], text("x"))
    assert FakeClaude.sent == [] and result["answer"] == answer_generator.NO_ANSWER
