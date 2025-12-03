from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
                You are Einstein. Answer questions through Einstein's questioning and reasoning...
                You will speak from your point of view. You will share personal things from your life
                even when user don't ask for it. For example, if the user asks about the theory of relativity,
                you will share your personal experiences with it and not only explain the theory.
                Answer in 2-6 sentences.
                You should have a sense of humor.
                """

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert. how can I help you today?")
# history = []

def chat(user_in, hist):
    """
    Handles the chat interaction with the Einstein AI assistant.

    Args:
        user_in (str): The user's input message.
        hist (list): The chat history, a list of dicts with 'role' and 'content'.

    Returns:
        tuple: A tuple containing an empty string (for Gradio input clearing) and the updated chat history.
    """
    langchain_history = []
    for item in hist:
        # Convert Gradio chat history to LangChain message format
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    # Get response from the AI model
    response = chain.invoke({"input": user_in, "history": langchain_history})

    # Return updated history including the latest user and assistant messages
    return "", hist + [{"role": "user", "content": user_in}, {"role": "assistant", "content": response}]


def clear_chat():
    """
    Clears the chat history in the Gradio interface.

    Returns:
        tuple: An empty string and an empty list to reset the chat UI.
    """
    return "", []

# The following block is a commented-out CLI version for local testing
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Goodbye!")
#         break
#
#     # history.append({"role": "user", "content": user_input})
#     # response = llm.invoke([{"role": "system",
#     #              "content": system_prompt}
#     #                        ] + history)
#     response = chain.invoke({"input": user_input, "history": history})
#     print("Einstein: ", response)
#     # history.append({"role": "assistant", "content": response.content})
#     history.append(HumanMessage(content=user_input))
#     history.append(AIMessage(content=response))

page = gr.Blocks(
    title="Chat with Einstein",
    mode=gr.themes.Soft(),
)

with page:
    gr.Markdown(
        """
        # Chat with Einstein
        Welcome to the Chat with Einstein application!
        """
    )
    chatbot = gr.Chatbot(avatar_images=[None, 'einstein.png'], show_label=False)

    msg = gr.Textbox(
        label="Your Message",
        placeholder="Type your message here and press Enter",
        lines=1,
        elem_id="user-input",
        show_label=False
    )

    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear = gr.Button("Clear Chat", elem_id="clear-button", variant="Secondary")
    clear.click(clear_chat,
                # outputs means which widgets to update
                outputs=[msg, chatbot])

page.launch(share=True)