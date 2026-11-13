"""
API failures must come back as a short, safe message — never as an uncaught
exception that crashes the page and shows a traceback in the browser.
Uses a fake client: no network, no real API key.
"""
import sys
from pathlib import Path

import anthropic
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import answer_generator

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
CHUNKS = [{"text": "Revenue grew 12.1%.", "doc_name": "a.pdf", "page": 1}]


def _status(cls, code):
    return cls("error", response=httpx.Response(code, request=REQUEST), body=None)


def _client_raising(error):
    class _Messages:
        def create(self, **kwargs):
            raise error

    class _Client:
        def __init__(self, *args, **kwargs):
            self.messages = _Messages()

    return _Client


@pytest.mark.parametrize("error, expected", [
    (anthropic.APIConnectionError(request=REQUEST), "couldn't reach the Claude API"),
    (anthropic.APITimeoutError(request=REQUEST), "couldn't reach the Claude API"),
    (_status(anthropic.AuthenticationError, 401), "rejected the API key"),
    (_status(anthropic.RateLimitError, 429), "rate limit"),
    (_status(anthropic.InternalServerError, 500), "having problems"),
    (_status(anthropic.NotFoundError, 404), "request to the Claude API failed"),
], ids=["connection", "timeout", "auth", "rate-limit", "server", "other"])
def test_api_errors_return_a_safe_message(monkeypatch, error, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(answer_generator.anthropic, "Anthropic", _client_raising(error))

    result = answer_generator.generate_answer("What was revenue growth?", CHUNKS)

    assert expected in result["answer"]
    assert result["sources"] == []
    assert "Traceback" not in result["answer"] and "\\" not in result["answer"]
