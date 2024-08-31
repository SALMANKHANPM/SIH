# import streamlit as st
# from langchain_community.llms import Ollama

# # Setting page title and header
# st.set_page_config(page_title="Student Assistance System")
# st.markdown("<h1>Student Assistance System</h1>", unsafe_allow_html=True)
# st.markdown("<p>By Department of Technical Education, Rajasthan</p>", unsafe_allow_html=True)
# st.divider()

# # Initialize session state
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [
#         {"role": "system", "content": "You're a Student Assistance System for assisting students with information regarding colleges, university, courses."}
#     ]
# if 'model_name' not in st.session_state:
#     st.session_state['model_name'] = []
# if 'user_input' not in st.session_state:
#     st.session_state['user_input'] = ""

# # Sidebar for model selection and conversation clearing
# st.sidebar.write("**Settings**")
# model_name = st.sidebar.radio("Choose a model:", ("tinyllama", "llama2"))
# clear_button = st.sidebar.button("Clear Conversation", key="clear")

# # Reset conversation
# if clear_button:
#     st.session_state['generated'] = []
#     st.session_state['past'] = []
#     st.session_state['messages'] = [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]
#     st.session_state['model_name'] = []
#     st.session_state['user_input'] = ""

# # Input container
# st.write("Or type your message here:")
# user_input = st.chat_input("Type your message here...")

# if user_input:
#     # Store the input in session state temporarily
#     st.session_state['user_input'] = user_input

#     # Generate response
#     llm = Ollama(model=model_name)
#     with st.spinner("Generating..."):
#         response = llm(user_input)
    
#     # Update session state with new messages
#     st.session_state['past'].append(user_input)
#     st.session_state['generated'].append(response)
#     st.session_state['model_name'].append(model_name)

#     # Clear the input field by resetting the session state
#     st.session_state['user_input'] = ""

#     # Display the new message
#     with st.chat_message("user"):
#         st.write(user_input)
#     with st.chat_message("assistant"):
#         st.write(response)
#     st.write(f"Model used: {model_name}")

# # Display chat history only if there are messages
# if st.session_state['generated']:
#     st.title("Chat History")
#     for i in range(len(st.session_state['generated'])):
#         with st.chat_message("user"):
#             st.write(st.session_state['past'][i])
#         with st.chat_message("assistant"):
#             st.write(st.session_state['generated'][i])
#         st.write(f"Model used: {st.session_state['model_name'][i]}")


