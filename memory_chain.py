from dotenv import load_dotenv
import os

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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




load_dotenv()

#LOad Gemini

# llm = ChatGoogleGenerativeAI(
#     model = "gemini-2.5-flash",
#     temperature = 0
# )

llm = ChatOpenAI(
    model = "deepseek/deepseek-chat",
    temperature=0,
    api_key = os.getenv("OPENROUTER_API_KEY"),
    base_url = "https://openrouter.ai/api/v1"
)

def create_rag_chain(retriever):
    history_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    qa_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    return create_retrieval_chain(
        history_retriever,
        qa_chain,
    )


def create_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(search_kwargs={"k": k})




