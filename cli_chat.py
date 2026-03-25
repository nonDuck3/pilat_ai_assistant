from src.doc_load_chunk import load_documents, split_text
from src.extract_store_embeddings import get_store_embeddings
from src.text_retrieval_search import query_vector_store
from src.query_rag_llm import generate_response_from_prompt
import argparse

import os
import getpass
from dotenv import load_dotenv

def rag_pipeline(pdf_file: str, tenant, db, chroma_api_key, collection_name: str, query_string: str, prompt_file: str):
    docs = load_documents(pdf_file)
    chunked_docs = split_text(docs)
    cloud_collection = get_store_embeddings(tenant, db, chroma_api_key, collection_name, chunked_docs)
    retrieved_docs = query_vector_store(cloud_collection, query_string)
    generate_response_from_prompt(prompt_file, retrieved_docs, query_string)

def main():
    load_dotenv()

    gemini_api_key = os.environ["GEMINI_API_KEY"]
    chroma_api_key = os.environ["CHROMADB_API_KEY"]
    tenant = os.environ["TENANT"]
    db = os.environ["DATABASE_NAME"]

    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    parser = argparse.ArgumentParser(description="RAG Pipeline: A tool to query local documents and generate answers using llama3.2.")
    parser.add_argument("pdf_file", type=str, help="The file path of the PDF that you want to process")
    parser.add_argument("collection_name", type=str, help="The name of the collection to store vector embeddings")
    parser.add_argument("query_string", type=str, help="The user query string")
    parser.add_argument("prompt_file", type=str, help="The prompt file given to the LLM")
    args = parser.parse_args()

    print(f"Received PDF file from user: {args.pdf_file}")
    print(f"Getting/creating collection: {args.collection_name}")
    print(f"Question: {args.query_string}")
    print(f"Prompt file provided by user: {args.prompt_file}")

    rag_pipeline(args.pdf_file,
        tenant=tenant,
        db=db,
        chroma_api_key=chroma_api_key,
        collection_name=args.collection_name,
        query_string=args.query_string,
        prompt_file=args.prompt_file
    )

if __name__ == "__main__":
    main()