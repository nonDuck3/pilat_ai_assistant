from src.extract_store_embeddings import get_vector_store_cloud_client
from src.text_retrieval_search import query_vector_store

import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

prompt_file = "prompts/system_prompt.md"

with open(prompt_file, 'r', encoding='utf-8') as file:
    base_system_prompt = file.read()

chroma_api_key = os.environ["CHROMADB_API_KEY"]
tenant = os.environ["TENANT"]
db = os.environ["DATABASE_NAME"]

client = get_vector_store_cloud_client(tenant, db, chroma_api_key)
collection = client.get_collection(name="pilates_guide_collection")

if "model" not in st.session_state:
    st.session_state["model"] = "llama3.2"

st.set_page_config(
    page_title="Pilat.ai",
    page_icon="🧘‍♀️"
)

st.title("🔮 Introducing Pilat.ai")
st.subheader("Elevate Your Pilates Training Journey with AI today.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🙆‍♀️" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

active_prompt = None

if "pilates_suggestions" in st.session_state and st.session_state.pilates_suggestions:
    active_prompt = st.session_state.pilates_suggestions

    # Reset the widget state immediately
    st.session_state.pilates_suggestions = None 

chat_input = st.chat_input("Ready? Set? Chat with Pilat.ai!")

if chat_input:
    active_prompt = chat_input

if active_prompt:

    with st.chat_message("user", avatar="🙆‍♀️"):
            st.markdown(active_prompt)
    st.session_state.messages.append({"role": "user", "content": active_prompt})

    retrieved_docs = query_vector_store(collection, active_prompt)
    context_text = "\n\n".join(retrieved_docs)
    
    # Provide context to system prompt via retrieved chunks from Chroma
    system_prompt = f"""{base_system_prompt}

    RELEVANT CONTEXT:
    {context_text}
    
    Please ensure you always answer the question based on the context provided above.
    """

    messages = [
        {"role": "user", "content": f"SYSTEM INSTRUCTIONS: {system_prompt}\n\nUser question: {active_prompt}"},
        *st.session_state.messages[:-1]  
    ]

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pilat.ai is thinking..."):
            placeholder = st.empty()
            response_text = ""
            
            try:
                response = requests.post(
                    os.getenv("OLLAMA_URL"),
                    json={
                        "model": st.session_state["model"],
                        "messages": messages,
                        "stream": True,
                    },
                    stream=True,
                )
                
                for line in response.iter_lines():
                    if line:
                        try:
                            # Decode and parse
                            chunk = json.loads(line.decode("utf-8"))
                            
                            # Check if 'done' is true (Ollama's final signal)
                            if chunk.get("done"):
                                break
                                
                            # Extract content safely
                            content = chunk.get("message", {}).get("content", "")
                            
                            if content:
                                response_text += content
                                # Update UI with a "typing" cursor
                                placeholder.markdown(response_text + "▌")
                        except json.JSONDecodeError:
                            continue # Skip malformed lines

                # Final render without the cursor
                placeholder.markdown(response_text)
        
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Ollama. Make sure it's running on http://localhost:11434")

    st.session_state.messages.append({"role": "assistant", "content": response_text})

if not st.session_state.messages:
    suggestions = [
        "What is the difference between a pelvis put in neutral and imprint?",
        "Explain the basic principles of STOTT Pilates.",
        "How can I get the benefits of Pilates by practicing STOTT?"
    ]
    st.pills("Suggestions", suggestions, key="pilates_suggestions", label_visibility="collapsed")
            
