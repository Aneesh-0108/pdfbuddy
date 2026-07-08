import gradio as gr

from memory_chain import(
    HumanMessage,
    AIMessage,
    rag_chain

)


chat_history = []


def chat(message,history):
    global chat_history
    result = rag_chain.invoke(
        {
            "input":message,
            "chat_history":chat_history
        }
    )
    
    answer = result["answer"]

    chat_history.append(
        HumanMessage(content=message)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    return answer 

main = gr.ChatInterface(
    chat,
    api_name="chat",
)

main.launch()