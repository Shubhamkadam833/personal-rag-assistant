"""
app.py — the main Streamlit application for the Personal AI Document Assistant.

WHAT THIS APP DOES:
  1. Upload PDFs → extract text → split into chunks → embed → store vectors on disk
  2. User asks a question → find matching chunks → send to Claude → show answer + sources

RUN WITH:
  streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent))

from rag.pdf_loader import load_pdf
from rag.text_chunker import chunk_pages
from rag.embeddings import embed_texts
from rag import vector_store
from rag.retriever import retrieve
from rag.answer_generator import generate_answer

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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

# ── Sidebar: Upload + Document Management ───────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload.",
    )

    if uploaded_files:
        for uf in uploaded_files:
            save_path = UPLOAD_DIR / uf.name
            save_path.write_bytes(uf.read())

            with st.spinner(f"Processing {uf.name}…"):
                try:
                    # 1. Extract text page by page
                    pages = load_pdf(save_path)

                    # 2. Split into smaller chunks
                    chunks = chunk_pages(pages, chunk_size=150, overlap=30)

                    # 3. Convert each chunk to an embedding vector
                    texts = [c["text"] for c in chunks]
                    embeddings = embed_texts(texts)

                    # 4. Store vectors on disk (duplicates are skipped automatically)
                    new_count = vector_store.add_chunks(chunks, embeddings)

                    if new_count > 0:
                        st.success(f"✅ {uf.name} — {new_count} chunks added")
                    else:
                        st.info(f"ℹ️ {uf.name} — already in the database, skipped")

                except ValueError as e:
                    st.error(f"❌ {uf.name}: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error processing {uf.name}: {e}")

    st.divider()

    # List loaded documents
    docs = vector_store.list_documents()
    if docs:
        st.subheader("Loaded documents")
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 {doc}")
            if col2.button("🗑", key=f"del_{doc}", help=f"Remove {doc}"):
                removed = vector_store.remove_document(doc)
                st.success(f"Removed {doc} ({removed} chunks deleted)")
                st.rerun()
    else:
        st.info("No documents loaded yet. Upload a PDF above.")

    st.divider()
    st.caption(f"Total chunks stored: {vector_store.total_chunks()}")

    if vector_store.total_chunks() > 0:
        if st.button("Rebuild index", help="Clear all stored chunks, then re-upload your PDFs."):
            cleared = vector_store.clear_all()
            st.success(f"Index cleared ({cleared} chunks). Re-upload your PDFs.")
            st.rerun()

# ── Main Area: Chat Interface ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(render_answer(msg["content"]))
        if msg.get("sources"):
            with st.expander("📌 Sources"):
                for s in msg["sources"]:
                    st.write(f"• **{s['doc_name']}** — page {s['page']}")

# Input
if question := st.chat_input("Ask a question about your documents…"):
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

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer…"):
                chunks = retrieve(question, top_k=5, history=history)
                result = generate_answer(question, chunks, history=history)

            st.markdown(render_answer(result["answer"]))

            if result["sources"]:
                with st.expander("📌 Sources"):
                    for s in result["sources"]:
                        st.write(f"• **{s['doc_name']}** — page {s['page']}")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })

# Clear chat button
if st.session_state.messages:
    if st.button("🗑 Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
