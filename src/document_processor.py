import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_documents_from_path(file_path):
    """
    Loads documents from a given file path with improved error handling and support for more file types.

    Args:
        file_path: Path to the document file

    Returns:
        List of document objects or empty list if loading fails
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return []

    try:
        # Dispatch to the appropriate loader based on file extension
        file_extension = os.path.splitext(file_path)[1].lower()

        loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }

        if file_extension in loaders:
            loader_class = loaders[file_extension]
            loader = loader_class(file_path)
            logger.info(f"Loading {file_path} with {loader_class.__name__}")
            docs = loader.load()

            if not docs:
                logger.warning(f"No content extracted from {file_path}")
                st.warning(f"No content could be extracted from {file_path}")
            else:
                logger.info(f"Successfully loaded {len(docs)} document segments from {file_path}")

            return docs
        else:
            supported_extensions = ', '.join(loaders.keys())
            st.warning(f"Unsupported file type: {file_extension}. Supported types: {supported_extensions}")
            return []

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}", exc_info=True)
        st.error(f"Error loading document {file_path}: {str(e)}")
        return []


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into chunks with configurable parameters.

    Args:
        docs: List of document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of text chunks or empty list if no documents provided
    """
    if not docs:
        logger.warning("No documents provided for splitting")
        return []

    try:
        logger.info(f"Splitting {len(docs)} documents with chunk size {chunk_size} and overlap {chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(docs)
        logger.info(f"Created {len(texts)} text chunks")
        return texts
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
        st.error(f"Error splitting documents: {str(e)}")
        return []