# Function to search existing documents in vector db
def query_vector_store(collection, query_string: str):
    """
    Perform a similarity search on the given vector store using a query string.

    Args:
        collection (str): The collection object.
        query_string (str): The query string used to find similar documents.

    Returns:
        list: A list of tuples, where each tuple contains a document and its 
              corresponding similarity score.
    """

    query_results = collection.query(
        query_texts=[query_string],
        include=["documents", "metadatas", "embeddings", "distances"],
        n_results=20
    )

    doc_text = query_results['documents'][0][0]
    metadata = query_results['metadatas'][0][0]
    score = query_results['distances'][0]
    doc_id = query_results['ids'][0][0]

    print(f"ID: {doc_id}")
    print(f"Score (Distance): {score}")
    print(f"Metadata: {metadata}")
    print(f"Content: {doc_text}")

    return doc_text
