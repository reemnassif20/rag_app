import streamlit as st
import os
import shutil
from src import config
from src.llm_handler import get_embeddings, get_llm
from src.document_processor import load_documents_from_path, split_documents
from src.vectorstore_handler import create_or_load_vectorstore, load_existing_vectorstore
from src.qa_chain_handler import get_qa_chain


# Initialize LLM and Embeddings
embeddings = get_embeddings()
llm = get_llm()

# Initialize session state for file uploader key if it doesn't exist
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = None


# Streamlit UI
st.title("📄 RAG Application")
st.markdown(f"Using LLM: `{config.OLLAMA_MODEL}`")

# File uploader
uploaded_files_current = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state.file_uploader_key}" # Use the key here
)


if uploaded_files_current:
    st.session_state.uploaded_files_cache = uploaded_files_current
    uploaded_files = uploaded_files_current
elif st.session_state.uploaded_files_cache and st.session_state.file_uploader_key == 0: # Only use cache if not cleared
     uploaded_files = st.session_state.uploaded_files_cache
else:
    uploaded_files = None


vectorstore = None

if uploaded_files:
    all_docs_content = []
    for uploaded_file in uploaded_files:
        # Ensure UPLOAD_DIR exists (it should be created by config.py, but good to double check)
        if not os.path.exists(config.UPLOAD_DIR):
            os.makedirs(config.UPLOAD_DIR)
            
        file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"Processing {uploaded_file.name}...")
        docs = load_documents_from_path(file_path)
        if docs:
            all_docs_content.extend(docs)
    
    if all_docs_content:
        with st.spinner("Splitting documents and updating vector store... This may take a while."):
            texts = split_documents(all_docs_content)
            if texts:
                vectorstore = create_or_load_vectorstore(texts, embeddings)
            else:
                st.warning("No text could be extracted from the uploaded documents.")
    else:
        st.warning("No processable documents found or an error occurred during loading.")


if not vectorstore: 
    vectorstore = load_existing_vectorstore(embeddings)

# Question Answering
if vectorstore:
    st.header("Ask a Question")
    query = st.text_input("Enter your question about the uploaded documents:", key="query_input")

    if st.button("Ask"):
        if query:
            with st.spinner("Thinking..."):
                qa_chain = get_qa_chain(llm, vectorstore)
                if qa_chain:
                    try:
                        result = qa_chain.invoke({"query": query})
                        st.subheader("Answer:")
                        st.write(result["result"])
                        
                        with st.expander("Show Source Documents"):
                            for doc in result["source_documents"]:
                                st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                                st.write(doc.page_content[:500] + "...") # Display first 500 chars
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"Error during question answering: {e}")
                else:
                    st.error("Could not initialize QA chain. Vectorstore might be empty.")
        else:
            st.warning("Please enter a question before clicking 'Ask'.")
else:
    st.info("Please upload documents to build or update the knowledge base, or ensure an existing vector store is present.")

st.sidebar.header("Settings")
st.sidebar.info(f"Ollama Model: {config.OLLAMA_MODEL}")
st.sidebar.info(f"Ollama URL: {config.OLLAMA_BASE_URL}")
st.sidebar.info(f"Vector Store Path: {config.VECTORSTORE_PATH}")


if st.sidebar.button("Clear Uploaded Files"):
    if os.path.exists(config.UPLOAD_DIR):
        try:
            shutil.rmtree(config.UPLOAD_DIR)
            st.sidebar.success("Uploaded files folder cleared.")
            # Recreate the directory after deleting it so future uploads don't fail
            os.makedirs(config.UPLOAD_DIR)
        except Exception as e:
            st.sidebar.error(f"Error clearing uploaded files folder: {e}")
    else:
        st.sidebar.info("Uploaded files folder not found, nothing to clear.")
    
    # Increment the key for the file uploader to reset it
    st.session_state.file_uploader_key += 1
    st.session_state.uploaded_files_cache = None # Clear cached files
    st.rerun()

# Button to clear vector store
if st.sidebar.button("Clear Vector Store"):
    if os.path.exists(config.VECTORSTORE_PATH):
        try:
            if os.path.isdir(config.VECTORSTORE_PATH):
                shutil.rmtree(config.VECTORSTORE_PATH)
            else: # if it's a file (should not happen with current FAISS saving)
                os.remove(config.VECTORSTORE_PATH)
            st.sidebar.success("Vector store cleared.")
        except Exception as e:
            st.sidebar.error(f"Error clearing vector store: {e}")
    else:
        st.sidebar.info("Vector store not found, nothing to clear.")
    st.rerun()