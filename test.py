import logging  # Import logging module
import json
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_experimental.text_splitter import SemanticChunker  # Import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama  # Import ChatOllama
import os
from concurrent.futures import ProcessPoolExecutor  # Import for parallel processing

# Add this line at the beginning of the file to define the store
store = {}  # Initialize the store as a dictionary

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Enable verbose logging
logger = logging.getLogger(__name__)

os.environ['USER_AGENT'] = 'myagent'

class CustomJSONLoader(JSONLoader):
    def _get_text(self, sample):
        # Check if the sample is a dictionary and has the expected structure
        if isinstance(sample, dict):
            # Modify this part based on your JSON structure
            if 'content' in sample:  # Example key
                return sample['content']  # Return the content
            else:
                logger.warning("Sample does not have the expected structure.")
                return json.dumps(sample)  # Fallback to dumping the whole sample
        return super()._get_text(sample)

def load_documents(directory: str) -> list:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(
        directory,
        glob="*.json",
        loader_cls=CustomJSONLoader,
        loader_kwargs={'jq_schema': '.'}
    )
    logger.debug("Loading data from directory...")
    data = loader.load()
    logger.debug(f"Loaded {len(data)} documents.")
    return data

def split_documents(data: list) -> list:
    """Split documents into manageable chunks using SemanticChunker."""
    text_splitter = SemanticChunker()  # Use SemanticChunker for semantic splitting
    splits = text_splitter.split_documents(data)
    logger.debug(f"Split into {len(splits)} chunks.")
    return splits

def create_vectorstore(splits: list, model_name: str, persist_directory: str) -> Chroma:
    """Create a vector store from the document splits."""
    embedding_function = OllamaEmbeddings(model=model_name, show_progress=True)
    try:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=persist_directory)
        logger.debug("Stored vectors in ChromaDB.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to store vectors in ChromaDB: {e}")
        raise

def create_rag_chain(vectorstore: Chroma, model_name: str) -> RunnableWithMessageHistory:
    """Create a RAG chain for question answering."""
    llm = ChatOllama(model=model_name)
    
    # Create the history-aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, vectorstore.as_retriever(), contextualize_q_prompt)

    # Create the question-answering chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    logger.debug("RAG chain created.")

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Manage chat history for sessions."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Main execution
if __name__ == "__main__":
    # Load documents
    data = load_documents('./dataset')

    # Split documents
    splits = split_documents(data)

    # Create vector store
    vectorstore = create_vectorstore(splits, model_name="mxbai-embed-large", persist_directory='db')

    # Create RAG chain
    conversational_rag_chain = create_rag_chain(vectorstore, model_name="llama3")

    # Invoke the chain and log the output
    try:
        logger.debug("Invoking conversational RAG chain...")
        response = conversational_rag_chain.invoke(
            {"input": "What are common ways of doing it?"},
            config={"configurable": {"session_id": "abc123"}},
        )
        logger.debug(f"Response: {response['answer']}")
        print(response["answer"])
    except Exception as e:
        logger.error(f"Error during invocation: {e}")