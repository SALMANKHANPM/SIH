import streamlit as st
from langchain_ollama import ChatOllama

# Importing Model
model = ChatOllama(model="tinyllama")

# App Title
st.set_page_config(page_title="Student Assistance System")

# Sidebar
with st.sidebar:
    st.image("streamlit_chat/Banner.jpg")
    st.title("Student Assistant System")
    st.write("By KL GLUG")
    st.write("A smart assistance to help students pursue their studies in dream colleges.")
    
    st.markdown("[Chat Now!](#chat)")
    st.link_button("View Source", url="https://github.com/")

# Storing History of Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you out today?"}]

# Display or Clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clearChatHistory():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you out today?"}]
st.sidebar.button('Clear Chat History', on_click=clearChatHistory)

# Prompt template
template = """<bos><start_of_turn>user\nYou are a Student Assistant System designed to help students with information regarding colleges, universities, courses, and their careers. \
Please provide a meaningful response based on the following context. Ensure your answer is in full sentences with correct spelling and punctuation. \
If the context does not contain the answer, respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model\n
ANSWER:"""

def generateResponse(promptInput):
    strDialogue = "You're a Student Assistance System for assisting students with information regarding colleges, universities, courses, and their careers."

    for dictMessage in st.session_state.messages:
        if dictMessage["role"] == "user":
            strDialogue += "User: " + dictMessage["content"] + "\n\n"
        else:
            strDialogue += "Assistant: " + dictMessage["content"] + "\n\n"

    # Prepare context and question for the template
    context = strDialogue
    question = promptInput
    prompt = template.format(context=context, question=question)

    # Generate response using the model
    output = model.invoke([("assistant", prompt)]).content
    return output

# User Prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new Response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            response = generateResponse(prompt)
            placeholder = st.empty()
            fullResponse = ''
            for item in response:
                fullResponse += item
                placeholder.markdown(fullResponse)
            placeholder.markdown(fullResponse)

        message = {"role": "assistant", "content": fullResponse}
        st.session_state.messages.append(message)