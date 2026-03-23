# Function to provide context to RAG and allow users to query LLM based on prompt
import requests

def generate_response_from_prompt(prompt_file: str, retrieved_docs, user_query: str):
    """
    Generate a response by combining a prompt template with retrieved documents and user query.
    
    Args:
        prompt_file (str): Path to the prompt template file.
        retrieved_docs: Retrieved documents or context to be used in response generation (list or iterable).
        user_query (str): The user's original query or question.
    
    Returns:
        str: The generated response based on the prompt template, retrieved documents, and user query.
    """
    with open(prompt_file, 'r', encoding='utf-8') as file:
        template = file.read()

    # Format the retrieved documents into one string
    context_text = "\n\n".join(retrieved_docs)

    # Inject into the template
    prompt = template.replace("{{context}}", context_text)
    final_prompt = prompt.replace("{{question}}", user_query)

    ollama_url = "http://localhost:11434/api/generate"
    model = "llama3.2"


    payload = {
        "model": model,
        "prompt": final_prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload)
    data = response.json()

    print("Response: \n", data["response"])