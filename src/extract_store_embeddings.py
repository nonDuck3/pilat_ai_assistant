import chromadb.utils.embedding_functions as embedding_functions
from tqdm import tqdm
import chromadb
from langchain_chroma import Chroma
import time

# Function to convert text to embeddings (Google Gemini Embedding model) and store embeddings in vector db (Chroma)
def get_store_embeddings(tenant: str, db: str, chroma_api_key: str, collection_name: str, chunked_doc, description: str = "Embedding + Uploading to Chroma"):
    """
    Store document embeddings in a Chroma vector database collection.
    
    Args:
        tenant (str): The tenant identifier for multi-tenant support.
        db (str): The database connection or client instance.
        chroma_api_key (str): API key for authenticating with Chroma.
        collection_name (str): Name of the Chroma collection to store embeddings in.
        chunked_doc: The document chunks to be embedded and stored (list or iterable).
        description (str): Description used to track upload progress of text embeddings.
    
    Returns:
        chromadb.api.models.Collection: The Chroma collection where the embeddings were stored.
    """

    # Explicit tenant and database
    client = get_vector_store_cloud_client(tenant, db, chroma_api_key)

    # get embedding function
    google_ef = get_embedding_model()

    # get/create a chroma collection in the cloud to store the embeddings
    chroma_collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=google_ef
    )

    # get page content and metadata from document chunks
    text_page_content = [d.page_content for d in chunked_doc]
    text_metadata = [d.metadata for d in chunked_doc]
    chunk_ids   = [f"text_chunk-{i}" for i in range(len(text_page_content))]

    # upload embeddings in batches per API call in chroma
    batch_size = 50
    for start in tqdm(range(0, len(text_page_content), batch_size), desc=description):
        end = start + batch_size
        chroma_collection.upsert(
            documents=text_page_content[start:end],    
            metadatas=text_metadata[start:end],  
            ids=chunk_ids[start:end]           
        )
        time.sleep(30)

    return chroma_collection

def initialize_vector_store(collection_name, embedding_function):
    """
    Initializes and returns a vector store collection.

    Args:
        collection_name (str): The name of the collection to create or connect to
                               in the vector store.
        embedding_function (callable): A function or object used to generate
                                       embeddings for the stored documents.

    Returns:
        VectorStore: An initialized vector store instance associated with the
                     specified collection.
    """

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    return vector_store

def get_vector_store_cloud_client(tenant: str, db: str, chroma_api_key: str):
    """
    Initializes and returns a cloud-based vector store client.

    Args:
        tenant (str): The tenant identifier for the cloud service account.
        db (str): The database or collection name to connect to.
        chroma_api_key (str): The API key for authenticating with the Chroma
                              cloud service.

    Returns:
        ChromaClient: An authenticated client instance connected to the specified
                      cloud vector store.

    """
    client = chromadb.CloudClient(
        tenant=tenant, 
        database=db,
        api_key=chroma_api_key
    ) 
    return client

def get_embedding_model(model_name: str = "models/gemini-embedding-001"):
    """
    Retrieves and returns an initialized embedding model.

    Returns:
        google_ef: An embedding model instance ready to generate vector
                        embeddings from text input.
    """

    # Use Chroma’s built-in Gemini embedding function (server-side call to Google API)
    google_ef = embedding_functions.GoogleGenaiEmbeddingFunction(
        model_name=model_name
    )

    return google_ef
