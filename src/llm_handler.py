import streamlit as st
import requests
import logging
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from .config import OLLAMA_MODEL, OLLAMA_BASE_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_ollama_availability():
    """
    Checks if Ollama server is reachable and the required model is available.

    Returns:
        Tuple of (is_available, message)
    """
    try:
        # Check if Ollama server is reachable
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)

        if response.status_code != 200:
            return False, f"Ollama server returned status code {response.status_code}"

        # Check if the required model is available
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]

        if OLLAMA_MODEL not in model_names:
            available_models = ", ".join(model_names[:5])
            more_text = f" and {len(model_names) - 5} more" if len(model_names) > 5 else ""
            return False, f"Model '{OLLAMA_MODEL}' not found. Available models: {available_models}{more_text}"

        return True, f"Ollama server is available and model '{OLLAMA_MODEL}' is ready"

    except requests.exceptions.RequestException as e:
        return False, f"Cannot connect to Ollama server at {OLLAMA_BASE_URL}: {str(e)}"


def get_model_info():
    """
    Gets information about the currently configured model.

    Returns:
        Dictionary with model information or None if unavailable
    """
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/show",
            params={"name": OLLAMA_MODEL},
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        return None

    except requests.exceptions.RequestException:
        return None


def get_embeddings():
    """
    Initializes and returns Ollama embeddings with improved error handling.

    Returns:
        OllamaEmbeddings object or None if initialization fails
    """
    is_available, message = check_ollama_availability()
    if not is_available:
        st.error(f"Embeddings initialization failed: {message}")
        st.info("Please ensure Ollama is running and the required model is available.")
        logger.error(f"Embeddings initialization failed: {message}")
        st.stop()

    try:
        logger.info(f"Initializing Ollama embeddings with model {OLLAMA_MODEL}")
        start_time = time.time()

        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            show_progress=True
        )

        # Test embeddings with a simple phrase
        test_embedding = embeddings.embed_query("Test embedding functionality")
        embedding_time = time.time() - start_time

        if test_embedding and len(test_embedding) > 0:
            logger.info(
                f"Embeddings initialized successfully in {embedding_time:.2f} seconds. Vector size: {len(test_embedding)}")
            return embeddings
        else:
            raise ValueError("Embeddings model returned empty vectors")

    except Exception as e:
        error_msg = f"Error initializing Ollama embeddings: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.info(f"Please ensure Ollama is running and model '{OLLAMA_MODEL}' is downloaded.")
        st.stop()


def get_llm():
    """
    Initializes and returns Ollama LLM with improved error handling.

    Returns:
        Ollama LLM object or None if initialization fails
    """
    try:
        logger.info(f"Initializing Ollama LLM with model {OLLAMA_MODEL}")

        # Get model info for better configuration
        model_info = get_model_info()

        # Configure parameters based on model capabilities
        temperature = 0.1  # Default conservative temperature

        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            top_p=0.9,
            num_ctx=2048,  # Context window
            timeout=120  # Increased timeout for larger documents
        )

        # Test LLM with a simple prompt
        test_response = llm.invoke("Respond with a single word: Hello")

        if test_response:
            logger.info(f"LLM initialized successfully. Test response: {test_response[:20]}...")
            return llm
        else:
            raise ValueError("LLM returned empty response")

    except Exception as e:
        error_msg = f"Error initializing Ollama LLM: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.info(f"Please ensure Ollama is running and model '{OLLAMA_MODEL}' is downloaded.")
        st.stop()