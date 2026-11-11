"""
app.py — the main Streamlit application for the Personal AI Document Assistant.

WHAT THIS APP DOES:
  1. Upload PDFs → extract text → split into chunks → embed → store vectors on disk
  2. User asks a question → find matching chunks → send to Claude → show answer + sources

RUN WITH:
  streamlit run app.py
"""

import hashlib
import logging
import streamlit as st
from pathlib import Path
import sys

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent))

# Check HTTPS certificates against the Windows certificate store (as browsers
# and pip do), not the certifi list that requests/httpx ship with. Networks
# that inspect HTTPS (antivirus web shields, office networks) use a certificate
# Windows trusts but certifi doesn't, which made every Claude and Hugging Face
# call fail with CERTIFICATE_VERIFY_FAILED.
import truststore
truststore.inject_into_ssl()

from rag.text_chunker import chunk_pages
from rag.embeddings import embed_texts
from rag import vector_store
from rag import document_registry
from rag.retriever import retrieve
from rag.answer_generator import MAX_HISTORY_TURNS, generate_answer
from security.pdf_validator import (
    PDFValidationError,
    safe_display_name,
    safe_storage_path,
    validate_pdf_bytes,
)
from security.pdf_sandbox import extract_pages_with_timeout
from security.prompt_guard import sanitize_chunk_text, scan_for_injection
from security.privacy_gate import needs_approval, privacy_levels_of

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def render_answer(text: str) -> str:
    """
    Escape '$' before display. Streamlit's markdown reads $...$ as LaTeX math,
    which silently eats the text between two dollar amounts.
    """
    return text.replace("$", r"\$")

st.set_page_config(
    page_title="Personal AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Personal AI Document Assistant")
st.caption("Upload PDFs, ask questions, get answers with sources.")

def _rebuild_index() -> int:
    """Clear chunks, the registry, and the uploaded files the registry tracks. Works on a damaged index."""
    try:
        for record in document_registry.list_all().values():
            (UPLOAD_DIR / Path(record["stored_path"]).name).unlink(missing_ok=True)
    except vector_store.IndexDamagedError:
        pass  # registry unreadable: we can't tell which uploads were ours, so leave them all
    cleared = vector_store.clear_all()
    document_registry.clear_all()
    return cleared


# ── Sidebar: Upload + Document Management ───────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")

    # A crash mid-save can leave the index unreadable. Offer a way out instead
    # of a traceback; nothing else on the page can work until it's rebuilt.
    try:
        vector_store.total_chunks()
        document_registry.list_all()
    except vector_store.IndexDamagedError:
        st.error(
            "The search index is damaged, probably by an interrupted save. "
            "Click **Rebuild index**, then upload your PDFs again."
        )
        if st.button("Rebuild index", key="rebuild_damaged"):
            _rebuild_index()
            st.rerun()
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            display_name = safe_display_name(uf.name)
            file_key = f"{uf.name}_{uf.size}"

            # getvalue() reads the full buffer without disturbing any read
            # position, so it's safe to call again on every rerun.
            data = uf.getvalue()
            doc_id = hashlib.sha256(data).hexdigest()

            # Whole-document duplicate detection: identical bytes mean this
            # is the same file, full stop — skip before any parsing, chunking
            # or embedding work happens at all.
            if document_registry.is_indexed(doc_id):
                existing = document_registry.get(doc_id)
                st.info(
                    f"ℹ️ {display_name} is identical to an already-loaded "
                    f"document ('{existing['display_name']}') — skipped."
                )
                continue

            with st.expander(f"📄 {display_name}", expanded=True):
                privacy = st.selectbox(
                    "Privacy level",
                    options=list(document_registry.PRIVACY_LEVELS),
                    index=list(document_registry.PRIVACY_LEVELS).index(document_registry.DEFAULT_PRIVACY),
                    key=f"privacy_{file_key}",
                    help=(
                        "Public — no restriction. "
                        "Personal — you'll be asked to approve sending its excerpts "
                        "to Claude each time it's used in an answer. "
                        "Restricted — never sent to Claude, under any circumstance."
                    ),
                )

                if st.button("Add to index", key=f"add_{file_key}"):
                    save_path = None
                    with st.spinner(f"Processing {display_name}…"):
                        try:
                            # Validate BEFORE writing anything, so a rejected
                            # upload never lands in the uploads directory.
                            validate_pdf_bytes(data, uf.name)

                            # Stored under a random name: a hostile filename
                            # cannot escape the uploads directory or overwrite
                            # an existing file.
                            save_path = safe_storage_path(UPLOAD_DIR)
                            save_path.write_bytes(data)

                            # Extraction runs in an isolated, time-limited
                            # process — see security/pdf_sandbox.py.
                            pages = extract_pages_with_timeout(save_path, display_name)
                            chunks = chunk_pages(pages, chunk_size=150, overlap=30)

                            # Sanitize text and scan for injection phrasing
                            # BEFORE it is ever placed in a prompt or stored.
                            flagged_patterns = set()
                            for chunk in chunks:
                                chunk["text"] = sanitize_chunk_text(chunk["text"])
                                flagged_patterns.update(scan_for_injection(chunk["text"]))
                                chunk["doc_id"] = doc_id

                            texts = [c["text"] for c in chunks]
                            embeddings = embed_texts(texts)
                            new_count = vector_store.add_chunks(chunks, embeddings)

                            document_registry.add(
                                doc_id=doc_id,
                                display_name=display_name,
                                privacy=privacy,
                                stored_path=save_path.name,
                                flagged=bool(flagged_patterns),
                            )

                            if flagged_patterns:
                                st.warning(
                                    f"⚠️ {display_name} contains phrasing that looks like "
                                    "a prompt-injection attempt. It was still indexed — "
                                    "the app treats document content as data, never as "
                                    "instructions — but you should know it's there."
                                )
                            st.success(f"✅ {display_name} — {new_count} chunks added ({privacy})")
                            st.rerun()

                        except PDFValidationError as e:
                            if save_path:
                                save_path.unlink(missing_ok=True)
                            st.error(f"❌ {e}")
                        except Exception:
                            # The browser gets a generic message (exception text
                            # can carry internal paths); the full traceback goes
                            # to the terminal so the failure can be diagnosed.
                            logger.exception("Indexing failed for %s", display_name)
                            if save_path:
                                save_path.unlink(missing_ok=True)
                            st.error(f"❌ Could not process {display_name}.")

    st.divider()

    # List loaded documents
    PRIVACY_ICON = {"public": "🌐", "personal": "🔒", "restricted": "⛔"}
    docs = vector_store.list_documents()
    legacy_count = sum(1 for d in docs if d["doc_id"] is None)

    if docs:
        st.subheader("Loaded documents")
        for doc in docs:
            record = document_registry.get(doc["doc_id"]) if doc["doc_id"] else None

            if record:
                badge = PRIVACY_ICON[record["privacy"]]
                if record.get("flagged"):
                    badge += " ⚠️"
            else:
                badge = "⚠️ excluded (pre-dates document IDs)"

            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {doc['doc_name']} {badge}")

            # No delete for legacy entries — there is no stored_path on record
            # to find the matching file on disk. Rebuild index clears them.
            if doc["doc_id"]:
                if col2.button("🗑", key=f"del_{doc['doc_id']}", help=f"Remove {doc['doc_name']}"):
                    removed = vector_store.remove_document(doc["doc_id"])
                    file_record = document_registry.remove(doc["doc_id"])
                    if file_record:
                        (UPLOAD_DIR / file_record["stored_path"]).unlink(missing_ok=True)
                    st.success(f"Removed {doc['doc_name']} ({removed} chunks + file deleted)")
                    st.rerun()

        if legacy_count:
            st.caption(
                f"⚠️ {legacy_count} document(s) pre-date privacy classification and "
                "are excluded from answers. Click **Rebuild index** below, then "
                "re-upload them."
            )
    else:
        st.info("No documents loaded yet. Upload a PDF above.")

    st.divider()
    st.caption(f"Total chunks stored: {vector_store.total_chunks()}")

    if vector_store.total_chunks() > 0 or document_registry.list_all():
        if st.button("Rebuild index", help="Clear all stored chunks, then re-upload your PDFs."):
            cleared = _rebuild_index()
            st.success(f"Index cleared ({cleared} chunks). Re-upload your PDFs.")
            st.rerun()

# ── Main Area: Chat Interface ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_personal" not in st.session_state:
    # Set when retrieval returned at least one "personal"-classified chunk:
    # {"question":, "chunks":, "history":}. The answer is NOT generated — and
    # no API call is made — until the user explicitly approves it below.
    st.session_state.pending_personal = None


def _show_answer(result: dict, used_privacy: list[str]) -> None:
    with st.chat_message("assistant"):
        st.markdown(render_answer(result["answer"]))
        if result["sources"]:
            with st.expander("📌 Sources"):
                for s in result["sources"]:
                    st.write(f"• **{s['doc_name']}** — page {s['page']}")
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        # Remembered so a later call that resends this answer as history is
        # gated the same way as the documents it came from.
        "used_privacy": used_privacy,
    })


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(render_answer(msg["content"]))
        if msg.get("sources"):
            with st.expander("📌 Sources"):
                for s in msg["sources"]:
                    st.write(f"• **{s['doc_name']}** — page {s['page']}")

# Approval gate: a Personal-classified answer is waiting on a yes/no click.
# Rendered before the chat input so it takes priority over asking a new question.
if st.session_state.pending_personal is not None:
    pending = st.session_state.pending_personal
    personal_docs = sorted({
        c["doc_name"] for c in pending["chunks"] if c.get("privacy") == "personal"
    })
    if personal_docs:
        st.warning(
            f"This answer would send excerpts from **Personal** document(s) "
            f"({', '.join(personal_docs)}) to the Claude API. Proceed?"
        )
    else:
        st.warning(
            "This answer would resend earlier replies that used **Personal** "
            "documents to the Claude API, as conversation context. Proceed?"
        )
    col1, col2 = st.columns(2)
    approve = col1.button("✅ Send to Claude", key="approve_personal")
    cancel = col2.button("❌ Don't send", key="cancel_personal")

    if approve:
        with st.spinner("Generating answer…"):
            result = generate_answer(pending["question"], pending["chunks"], history=pending["history"])
        # Tagged personal even if only the history was: this answer was built
        # with Personal content in its context.
        used = sorted(set(privacy_levels_of(pending["chunks"])) | {"personal"})
        _show_answer(result, used)
        st.session_state.pending_personal = None
        st.rerun()
    elif cancel:
        _show_answer({
            "answer": "Okay — I won't send that Personal data to Claude. Ask something else, or mark the document Public if you'd like it answered automatically.",
            "sources": [],
        }, [])
        st.session_state.pending_personal = None
        st.rerun()

# Input — only offered once any pending approval is resolved.
elif question := st.chat_input("Ask a question about your documents…"):
    if vector_store.total_chunks() == 0:
        st.warning("Please upload at least one PDF document first.")
    else:
        # Capture the conversation so far BEFORE adding the current question,
        # so the question is not part of its own history.
        history = list(st.session_state.messages)

        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(render_answer(question))

        with st.spinner("Searching documents…"):
            chunks = retrieve(question, top_k=5, history=history)

        # Only the turns that will actually be resent (see trim_history).
        if needs_approval(chunks, history[-MAX_HISTORY_TURNS:]):
            # Stop here. No answer is generated and no API call is made until
            # the user explicitly approves on the next run, above.
            st.session_state.pending_personal = {
                "question": question,
                "chunks": chunks,
                "history": history,
            }
            st.rerun()
        else:
            with st.spinner("Generating answer…"):
                result = generate_answer(question, chunks, history=history)
            _show_answer(result, privacy_levels_of(chunks))

# Clear chat button
if st.session_state.messages:
    if st.button("🗑 Clear chat", key="clear_chat"):
        st.session_state.messages = []
        # A pending approval belongs to the conversation being cleared;
        # approving it later would resend the history the user just wiped.
        st.session_state.pending_personal = None
        st.rerun()
