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
You are a question answering asistant.

Answer ONLY using the information contained in the provided context.

If the answer cannot be found in the context,respond exactly with :

"I could not find this information in the provided document."

Do NOT use your own knowledge .
Do NOT guess.
DO NOT infer facts that are not explicitly stated

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