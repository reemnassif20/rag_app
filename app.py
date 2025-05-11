import streamlit as st
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredPowerPointLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
CONFIG_FILE = "config.yaml"


def load_config():
    """Load configuration from YAML file or use defaults"""
    DEFAULT_CONFIG = {
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "qwen3:0.6b",
            "temperature": 0.1
        },
        "storage": {
            "vectorstore_path": "vectorstore.faiss",
            "upload_dir": "uploaded_files"
        },
        "document_processing": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "retrieval": {
            "k_documents": 4
        }
    }

    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                return yaml.safe_load(file)
        else:
            # Create default config file
            with open(CONFIG_FILE, 'w') as file:
                yaml.dump(DEFAULT_CONFIG, file, default_flow_style=False)
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return DEFAULT_CONFIG


# Load config
config = load_config()

# Extract configuration values
OLLAMA_BASE_URL = config["ollama"]["base_url"]
OLLAMA_MODEL = config["ollama"]["model"]
VECTORSTORE_PATH = config["storage"]["vectorstore_path"]
UPLOAD_DIR = config["storage"]["upload_dir"]
CHUNK_SIZE = config["document_processing"]["chunk_size"]
CHUNK_OVERLAP = config["document_processing"]["chunk_overlap"]
K_DOCUMENTS = config["retrieval"]["k_documents"]

# Ensure required directories exist
for directory in [UPLOAD_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None


# Initialize AI components
def initialize_components():
    """Initialize and cache AI components"""
    # Return cached components if already initialized
    if st.session_state.embeddings and st.session_state.llm:
        return st.session_state.embeddings, st.session_state.llm

    with st.spinner("Initializing AI models..."):
        try:
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL
            )

            # Test embeddings with a simple query
            test_embedding = embeddings.embed_query("Test")
            if not test_embedding:
                raise ValueError("Embeddings initialization failed")

            # Initialize LLM
            llm = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            )

            # Cache for future use
            st.session_state.embeddings = embeddings
            st.session_state.llm = llm

            return embeddings, llm
        except Exception as e:
            st.error(f"Failed to initialize AI: {str(e)}")
            st.info("Please ensure Ollama is running with the correct model.")
            st.stop()


# Document processing
def load_document(file_path):
    """Load a document based on its file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.md': UnstructuredMarkdownLoader
    }

    if file_extension in loaders:
        try:
            loader = loaders[file_extension](file_path)
            return loader.load()
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return []
    else:
        st.warning(f"Unsupported file type: {file_extension}")
        return []


def process_documents(uploaded_files):
    """Process uploaded documents and create vectorstore"""
    if not uploaded_files:
        return

    # Initialize components
    embeddings, _ = initialize_components()

    # Process progress bar
    progress_bar = st.progress(0)
    status = st.empty()
    status.info("Processing documents...")

    all_docs = []
    file_count = len(uploaded_files)

    # Save and process each file
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i / file_count) * 0.5
        progress_bar.progress(progress)

        # Save file
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load document
        docs = load_document(file_path)
        all_docs.extend(docs)

    if not all_docs:
        status.error("No content extracted from documents")
        return

    # Split documents into chunks
    progress_bar.progress(0.6)
    status.info("Creating document chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(all_docs)

    if not texts:
        status.error("Failed to create text chunks")
        return

    # Create or update vectorstore
    progress_bar.progress(0.8)
    status.info("Building vector database...")

    try:
        if os.path.exists(VECTORSTORE_PATH) and st.session_state.vectorstore:
            # Add to existing vectorstore
            vectorstore = st.session_state.vectorstore
            vectorstore.add_documents(texts)
        else:
            # Create new vectorstore
            vectorstore = FAISS.from_documents(texts, embeddings)

        # Save vectorstore
        vectorstore.save_local(VECTORSTORE_PATH)
        st.session_state.vectorstore = vectorstore

        # Update progress and status
        progress_bar.progress(1.0)
        status.success(f"Processing complete! Added {len(texts)} document chunks.")

        # Clear cached QA chain to rebuild with new vectorstore
        st.session_state.qa_chain = None

        return vectorstore
    except Exception as e:
        status.error(f"Error creating vectorstore: {str(e)}")
        return None


def load_vectorstore():
    """Load existing vectorstore if available"""
    if st.session_state.vectorstore:
        return st.session_state.vectorstore

    if not os.path.exists(VECTORSTORE_PATH):
        return None

    embeddings, _ = initialize_components()
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = vectorstore
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None


def create_qa_chain():
    """Create QA chain for document retrieval and answering"""
    if st.session_state.qa_chain:
        return st.session_state.qa_chain

    embeddings, llm = initialize_components()
    vectorstore = st.session_state.vectorstore

    if not vectorstore:
        vectorstore = load_vectorstore()
        if not vectorstore:
            return None

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": K_DOCUMENTS}
    )

    # Create prompt template
    prompt_template = """
    You are an assistant that provides helpful answers based on the given context.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    st.session_state.qa_chain = qa_chain
    return qa_chain


def answer_question(question):
    """Process a question and return the answer"""
    qa_chain = create_qa_chain()
    if not qa_chain:
        st.error("Please upload documents first")
        return None

    with st.spinner("Searching documents..."):
        try:
            result = qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

            # Add to chat history
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

            return answer, sources
        except Exception as e:
            st.error(f"Error answering question: {str(e)}")
            return None, []


# UI elements
def render_sidebar():
    """Render sidebar with controls and status"""
    with st.sidebar:
        st.title("Document Q&A")

        # Show stats
        st.subheader("Status")
        col1, col2 = st.columns(2)
        with col1:
            doc_count = len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
            st.metric("Documents", doc_count)
        with col2:
            db_status = "Ready" if os.path.exists(VECTORSTORE_PATH) else "None"
            st.metric("Vector DB", db_status)

        # Show model info
        st.subheader("Model")
        st.code(OLLAMA_MODEL)

        # Management buttons
        st.subheader("Management")

        if st.button("Clear Files", key="clear_files"):
            if os.path.exists(UPLOAD_DIR):
                for file in os.listdir(UPLOAD_DIR):
                    os.remove(os.path.join(UPLOAD_DIR, file))
                st.session_state.file_uploader_key += 1
                st.success("Files cleared")
                st.rerun()

        if st.button("Clear Vector DB", key="clear_vectordb"):
            if os.path.exists(VECTORSTORE_PATH):
                if os.path.isdir(VECTORSTORE_PATH):
                    import shutil
                    shutil.rmtree(VECTORSTORE_PATH)
                else:
                    os.remove(VECTORSTORE_PATH)
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.success("Vector DB cleared")
                st.rerun()

        if st.button("Reset Chat", key="reset_chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared")
            st.rerun()


def render_upload_section():
    """Render document upload section"""
    st.header("Upload Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or other supported files",
        type=["pdf", "docx", "txt", "csv", "md", "html"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}"
    )

    if uploaded_files:
        # Process button
        if st.button("Process Documents", type="primary"):
            process_documents(uploaded_files)


def render_chat_section():
    """Render chat interface"""
    st.header("Ask Questions")

    # Initialize vectorstore if needed
    vectorstore = load_vectorstore()
    if not vectorstore:
        st.info("Please upload and process documents first")
        return

    # Display chat history
    for exchange in st.session_state.chat_history:
        # Question
        st.markdown(f"**Question:** {exchange['question']}")

        # Answer
        st.markdown(f"**Answer:** {exchange['answer']}")

        # Sources
        with st.expander("View Sources"):
            for i, doc in enumerate(exchange['sources']):
                source = doc.metadata.get('source', 'Unknown source')
                st.markdown(f"**Source {i + 1}:** {os.path.basename(source)}")
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

        st.divider()

    # Question input
    question = st.text_input("Enter your question:")

    if st.button("Ask", type="primary") and question:
        answer, sources = answer_question(question)
        if answer:
            st.rerun()  # Refresh to show new chat history


# Main application
def main():
    """Main application flow"""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main sections
    render_upload_section()
    st.divider()
    render_chat_section()


if __name__ == "__main__":
    main()