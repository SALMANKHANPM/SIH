import logging
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# Main execution
if __name__ == "__main__":
    # Load documents
    documents = load_documents('./dataset')

    # Create embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)

    # Create Semantic Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        add_start_index=True,
    )

    # Split documents into chunks
    texts = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory="./db-hormozi"
    )

    logger.debug("Vectorstore created")