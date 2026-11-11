"""
App-level test (Streamlit's AppTest): clearing the chat must also discard a
pending Personal-data approval. Otherwise approving it afterwards would send
the old question plus the history the user just cleared.

No API call is made: the approval is never approved in this test.
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "app.py")
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_clear_chat_discards_pending_approval():
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["messages"] = [{"role": "user", "content": "What is my salary?"}]
    at.session_state["pending_personal"] = {
        "question": "What is my salary?",
        "chunks": [{"text": "Salary 90k", "doc_name": "payslip.pdf", "page": 1, "privacy": "personal"}],
        "history": [{"role": "assistant", "content": "earlier private answer"}],
    }
    at.run()
    assert any("Personal" in w.value for w in at.warning), "approval prompt should be showing"

    at.button(key="clear_chat").click().run()
    # The Clear chat handler sits at the bottom of app.py, so the click's run
    # drew the prompt before clearing it; its st.rerun() then redraws the page.
    # AppTest doesn't follow that rerun on its own, so redraw like the browser.
    at.run()

    assert at.session_state["pending_personal"] is None
    assert at.session_state["messages"] == []
    assert not any("Personal" in w.value for w in at.warning), "approval prompt should be gone"
