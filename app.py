import streamlit as st

from memory_chain import AIMessage, HumanMessage, rag_chain


st.set_page_config(page_title="PDFBuddy", page_icon="📄")
st.title("PDFBuddy")
st.caption("Ask questions about the loaded PDF.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

prompt = st.chat_input("Ask a question about the PDF")

if prompt:
    result = rag_chain.invoke(
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