"""
query_rewriter.py — turns a follow-up question into a standalone one.

WHY THIS EXISTS:
  Search works by embedding the question and comparing it to stored chunks.
  A follow-up like "what about the prior year?" contains no words from the
  document, so its embedding matches nothing useful and retrieval fails —
  even though a human reading the conversation knows exactly what it means.

  So before searching, we rewrite the question using the conversation so far:
      "what about the prior year?"
        -> "What was Northstar Meridian's revenue growth in 2024?"

  The rewritten question is used ONLY for searching. The user's original
  wording is what gets sent to Claude for the final answer.

COST:
  Uses Haiku, the cheapest model, with a tiny prompt and a short reply.
  First questions in a conversation skip the call entirely.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You rewrite follow-up questions so they can stand alone.

Given a conversation and the user's latest question, rewrite the latest question
so it makes sense with no conversation context — resolve pronouns and references
like "it", "that", "the prior year", "those figures".

Rules:
1. Output ONLY the rewritten question. No preamble, no explanation, no quotes.
2. If the question already stands alone, output it unchanged.
3. Never answer the question. Never invent details that are not in the conversation.
4. Keep it short and keep the user's intent exactly."""


def condense_question(question: str, history: list[dict] | None = None) -> str:
    """
    Rewrite `question` into a standalone question using prior turns.

    Args:
        question: The user's latest question.
        history: Prior turns as dicts with "role" and "content".

    Returns:
        A standalone question. Falls back to the original question whenever
        rewriting is unnecessary or unavailable — searching with the original
        is always better than failing the request.
    """
    if not history:
        return question

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return question

    transcript = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}" for turn in history
    )
    user_message = f"""Conversation so far:
{transcript}

Latest question: {question}

Rewritten standalone question:"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL,
            # Generous for a one-line answer: Haiku thinks by default, and
            # thinking shares this budget. Too small a cap spends the whole
            # budget on thinking and returns no text at all.
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError:
        # Network failure, bad key, rate limit — searching with the user's
        # original wording is a worse search, but still a working app.
        return question

    # A truncated reply is a half-written question; the original is safer.
    if response.stop_reason == "max_tokens":
        return question

    for block in response.content:
        if block.type == "text" and block.text.strip():
            return block.text.strip()

    return question
