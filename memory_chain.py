from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import (
      create_history_aware_retriever,
      create_retrieval_chain
)
  
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import(
    HumanMessage,
    AIMessage
)

from prompts import(
    contextualize_q_prompt,
    qa_prompt
)




chat_history = []

load_dotenv()

#LOad Gemini

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0
)

#Load FAISS :

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True

)

# fetches top 4 results 
retriever = db.as_retriever(
    search_kwargs={"k":4}
)


#history-aware -retriever for more accuracy:

history_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt

)


qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt

)


rag_chain = create_retrieval_chain(
    history_retriever,
    qa_chain
)



while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": chat_history
        }
    )

    answer = result["answer"]

    print("\nBuddy:",answer)

    #saving user message and ai response to the context i.e to chat_history=[]

    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )




