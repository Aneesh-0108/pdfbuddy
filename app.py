from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from indexing import create_vectorstore_from_pdf
from langchain_core.messages import AIMessage, HumanMessage
from memory_chain import create_rag_chain, create_retriever


st.set_page_config(page_title="PDFBuddy", page_icon="📄")
st.title("PDFBuddy")
st.caption("Upload one PDF, then ask questions about that document.")


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "document_hash" not in st.session_state:
        st.session_state.document_hash = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "document_ready" not in st.session_state:
        st.session_state.document_ready = False
    if "document_status" not in st.session_state:
        st.session_state.document_status = "Upload a PDF to begin."
    if "document_status_type" not in st.session_state:
        st.session_state.document_status_type = "info"


def compute_file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def build_chain_from_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = Path(temp_file.name)

    try:
        vectorstore = create_vectorstore_from_pdf(temp_path)
        retriever = create_retriever(vectorstore)
        return create_rag_chain(retriever)
    finally:
        temp_path.unlink(missing_ok=True)


initialize_session_state()

uploaded_pdf = st.file_uploader("Upload a single PDF", type=["pdf"])

if uploaded_pdf is not None:
    uploaded_hash = compute_file_hash(uploaded_pdf)
    needs_indexing = (
        uploaded_hash != st.session_state.document_hash
        or st.session_state.rag_chain is None
    )

    if needs_indexing:
        st.info(f"Processing {uploaded_pdf.name}...")
        previous_ready = st.session_state.document_ready
        try:
            with st.spinner("Indexing uploaded PDF..."):
                st.session_state.rag_chain = build_chain_from_uploaded_pdf(uploaded_pdf)

            st.session_state.chat_history = []
            st.session_state.document_hash = uploaded_hash
            st.session_state.document_name = uploaded_pdf.name
            st.session_state.document_ready = True
            st.session_state.document_status = f"Indexed {uploaded_pdf.name}."
            st.session_state.document_status_type = "success"
        except Exception as error:
            st.session_state.document_ready = previous_ready
            st.session_state.document_status = (
                f"Could not index {uploaded_pdf.name}: {error}"
            )
            st.session_state.document_status_type = "error"
    elif st.session_state.document_ready:
        st.session_state.document_status = f"Using {st.session_state.document_name}."
        st.session_state.document_status_type = "success"

status_type = st.session_state.document_status_type
status_message = st.session_state.document_status

if status_type == "success":
    st.success(status_message)
elif status_type == "error":
    st.error(status_message)
else:
    st.info(status_message)

if st.session_state.document_name:
    st.caption(f"Active document: {st.session_state.document_name}")

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

prompt = st.chat_input(
    "Ask a question about the PDF",
    disabled=not st.session_state.document_ready,
)

if prompt and st.session_state.rag_chain is not None:
    result = st.session_state.rag_chain.invoke(
        {
            "input": prompt,
            "chat_history": st.session_state.chat_history,
        }
    )

    answer = result["answer"]

    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.write(answer)