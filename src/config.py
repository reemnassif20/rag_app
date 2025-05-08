import os

OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "qwen3:4b"  # Your downloaded model
VECTORSTORE_PATH = "vectorstore.faiss"
UPLOAD_DIR = "uploaded_files"

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)