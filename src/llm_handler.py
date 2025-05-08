import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from .config import OLLAMA_MODEL, OLLAMA_BASE_URL

def get_embeddings():
    """Initializes and returns Ollama embeddings."""
    try:
        return OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        st.error(f"Error initializing Ollama embeddings. Is Ollama running and the model '{OLLAMA_MODEL}' downloaded? Error: {e}")
        st.stop()

def get_llm():
    """Initializes and returns Ollama LLM."""
    try:
        return Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        st.error(f"Error initializing Ollama LLM. Is Ollama running and the model '{OLLAMA_MODEL}' downloaded? Error: {e}")
        st.stop()