from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Function to load PDF documents
def load_documents(file_path: str):
    """
    Load a PDF document from the specified file path using PyPDFLoader.

    Args:
        file_path (str): The file path to the PDF file.

    Returns:
        list: A list of documents loaded from the specified PDF file.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print("Length of document", len(docs))
    return docs

# Function to split and chunk loaded documents
def split_text(docs):
    """
    Splits a list of documents into smaller chunks using a RecursiveCharacterTextSplitter.

    Args:
        docs (list): A list of documents to be split. Each document should be in a format 
                     compatible with the RecursiveCharacterTextSplitter.

    Returns:
        list: A list of document chunks. Each chunk includes its start index if 
              `add_start_index` is set to True in the splitter configuration.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # get the number of splits (elements) for the pdf document 
    # if the length is 300 then it means that there are 300 elements in the list
    print(len(all_splits))
    return all_splits