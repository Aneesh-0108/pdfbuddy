from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Refining the prompt(Contextualization - Editor)



contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and latest user questions,rewrite the question "
            "so that it could be understood without the access of the chat history. "
            "Do NOT answer it."
        ),
        #creating a varibale - "chat_history"
        MessagesPlaceholder("chat_history"),
        ('human', "{input}")
    ]

)

#The EXPERT :

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Use the retrieved context to answer the questions.

If the answer is not found in the context,
say you do NOT know.

Context:

-------------------

{context}

-------------------

"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)