"""
privacy_gate.py — decides whether a Claude call needs the user's approval.

WHY THIS EXISTS:
  Approval used to be checked only against the chunks retrieved for the
  CURRENT question. But every call also resends earlier answers as chat
  history, and an earlier answer may contain facts taken from a Personal
  document. So a follow-up that only hits Public chunks could still resend
  Personal-derived text without asking. This module checks both.

  Kept free of Streamlit so it can be unit tested directly.
"""


def privacy_levels_of(chunks: list[dict]) -> list[str]:
    """Privacy levels present in a set of retrieved chunks, for tagging the answer."""
    return sorted({c["privacy"] for c in chunks if c.get("privacy")})


def needs_approval(chunks: list[dict], history: list[dict]) -> bool:
    """
    True if this call would send Personal content to Claude — either in the
    chunks retrieved now, or inside an earlier answer being resent as history.
    """
    if any(c.get("privacy") == "personal" for c in chunks):
        return True
    return any("personal" in turn.get("used_privacy", ()) for turn in history)
