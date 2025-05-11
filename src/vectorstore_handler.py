import streamlit as st
import os
import logging
import time
from datetime import datetime
from langchain_community.vectorstores import FAISS
from .config import VECTORSTORE_PATH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_backup(vectorstore_path):
    """
    Creates a backup of the existing vectorstore before modifying it.

    Args:
        vectorstore_path: Path to the vectorstore
    """
    if not os.path.exists(vectorstore_path):
        return

    backup_dir = os.path.join(os.path.dirname(vectorstore_path), "backups")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"vectorstore_backup_{timestamp}")

    try:
        import shutil
        shutil.copytree(vectorstore_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    except Exception as e:
        logger.warning(f"Failed to create backup: {str(e)}")


def create_or_load_vectorstore(texts, embeddings, backup=True):
    """
    Creates a new vector store or loads and updates an existing one with improved
    error handling and optional backup functionality.

    Args:
        texts: List of text chunks to add to the vectorstore
        embeddings: Embeddings model to use
        backup: Whether to create a backup before updating (default: True)

    Returns:
        FAISS vectorstore object or None if operation fails
    """
    if not texts:
        st.warning("No texts provided to create or update vector store.")
        return None

    # Create a progress bar
    progress_bar = st.progress(0)
    status_message = st.empty()

    try:
        if os.path.exists(VECTORSTORE_PATH):
            status_message.info("Loading existing vector store...")
            progress_bar.progress(0.2)

            if backup:
                create_backup(VECTORSTORE_PATH)

            try:
                # Load existing vectorstore
                start_time = time.time()
                vectorstore = FAISS.load_local(
                    VECTORSTORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                load_time = time.time() - start_time
                logger.info(f"Loaded existing vectorstore in {load_time:.2f} seconds")
                progress_bar.progress(0.5)

                # Add new documents
                status_message.info("Adding new documents to existing vector store...")
                start_time = time.time()
                vectorstore.add_documents(texts)
                add_time = time.time() - start_time
                logger.info(f"Added {len(texts)} documents in {add_time:.2f} seconds")
                progress_bar.progress(0.8)

                # Save updated vectorstore
                vectorstore.save_local(VECTORSTORE_PATH)
                progress_bar.progress(1.0)
                status_message.success(f"Vector store updated with {len(texts)} new documents!")

            except Exception as e:
                logger.error(f"Error loading or updating vector store: {str(e)}", exc_info=True)
                status_message.error(f"Error with existing vector store: {str(e)}. Creating a new one.")
                progress_bar.progress(0.5)

                # Create new vectorstore if loading fails
                vectorstore = FAISS.from_documents(texts, embeddings)
                vectorstore.save_local(VECTORSTORE_PATH)
                progress_bar.progress(1.0)
                status_message.success("New vector store created and saved!")
        else:
            # Create new vectorstore
            status_message.info("Creating new vector store...")
            progress_bar.progress(0.3)

            start_time = time.time()
            vectorstore = FAISS.from_documents(texts, embeddings)
            create_time = time.time() - start_time
            logger.info(f"Created vectorstore with {len(texts)} documents in {create_time:.2f} seconds")
            progress_bar.progress(0.8)

            # Save vectorstore
            vectorstore.save_local(VECTORSTORE_PATH)
            progress_bar.progress(1.0)
            status_message.success("New vector store created and saved!")

        return vectorstore

    except Exception as e:
        logger.error(f"Unexpected error in vectorstore operations: {str(e)}", exc_info=True)
        status_message.error(f"Error creating or updating vector store: {str(e)}")
        progress_bar.progress(1.0)
        return None


def load_existing_vectorstore(embeddings):
    """
    Loads an existing vector store if available with improved error handling.

    Args:
        embeddings: Embeddings model to use

    Returns:
        FAISS vectorstore object or None if loading fails
    """
    if os.path.exists(VECTORSTORE_PATH):
        try:
            status = st.empty()
            status.info("Loading existing vector store...")

            start_time = time.time()
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            load_time = time.time() - start_time

            # Get vectorstore stats
            doc_count = len(vectorstore.docstore._dict)

            status.success(f"Loaded vector store with {doc_count} documents in {load_time:.2f} seconds")
            logger.info(f"Successfully loaded vector store with {doc_count} documents")

            return vectorstore

        except Exception as e:
            logger.error(f"Error loading existing vector store: {str(e)}", exc_info=True)
            st.error(f"Error loading existing vector store: {str(e)}")
            return None

    logger.info("No existing vector store found")
    return None
