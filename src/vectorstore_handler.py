import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from .config import VECTORSTORE_PATH

def create_or_load_vectorstore(texts, embeddings):
    """Creates a new vector store or loads and updates an existing one."""
    if not texts:
        st.warning("No texts provided to create or update vector store.")
        return None

    if os.path.exists(VECTORSTORE_PATH):
        st.info("Loading existing vector store...")
        try:
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.info("Adding new documents to existing vector store...")
            vectorstore.add_documents(texts)
            vectorstore.save_local(VECTORSTORE_PATH)
            st.success("Vector store updated successfully!")
        except Exception as e:
            st.error(f"Error loading or updating vector store: {e}. Creating a new one from all provided texts.")
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(VECTORSTORE_PATH)
            st.success("New vector store created and saved!")
    else:
        st.info("Creating new vector store...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        st.success("New vector store created and saved!")
    return vectorstore

def load_existing_vectorstore(embeddings):
    """Loads an existing vector store if available."""
    if os.path.exists(VECTORSTORE_PATH):
        try:
            st.info("Loading existing vector store...")
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            st.success("Existing vector store loaded.")
            return vectorstore
        except Exception as e:
            st.error(f"Error loading existing vector store: {e}")
            return None
    return None