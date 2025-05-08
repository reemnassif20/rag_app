import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_path(file_path):
    """Loads documents from a given file path."""
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file_path}. Skipping.")
            return []
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document {file_path}: {e}")
        return []

def split_documents(docs):
    """Splits documents into chunks."""
    if not docs:
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts