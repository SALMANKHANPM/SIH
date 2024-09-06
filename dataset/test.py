import json
from langchain_community.document_loaders import DirectoryLoader, JSONLoader
import os

# Set USER_AGENT environment variable
os.environ['USER_AGENT'] = 'myagent'

# Custom JSONLoader to convert dict to string
class CustomJSONLoader(JSONLoader):
    def _get_text(self, sample):
        if isinstance(sample, dict):
            return json.dumps(sample)  # Convert the dict to a string
        return super()._get_text(sample)

# Initialize directory loader with CustomJSONLoader
directory_loader = DirectoryLoader(
    './',
    glob="*.json",
    loader_cls=CustomJSONLoader,
    loader_kwargs={'jq_schema': '.'}  # jq_schema set to select all content
)

# Load JSON files
documents = directory_loader.load()

# Process the documents as needed
for doc in documents:
    print(doc.page_content)  # This will now print the content as a JSON string
