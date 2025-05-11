# DocumentQA: Streamlined Document Intelligence

DocumentQA is a lightweight, efficient application that lets you ask questions about your documents using AI. Built with Streamlit, LangChain, and Ollama, it provides a simple interface for document-based question answering.

## Features

- **Simple Document Processing**: Upload PDFs, Word docs, text files, and more
- **Fast Question Answering**: Get AI-powered answers based on your documents
- **Local Privacy**: All processing happens on your machine
- **Lightweight**: Optimized for performance and efficiency
- **Source Citations**: See which documents contain the information

## Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) - For running the AI model locally

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/reemnassif02/documentqa.git
   cd documentqa
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the required model with Ollama**

   ```bash
   ollama pull qwen3:0.6b
   ```
   
   Note: You can use other models by updating the `config.yaml` file.

### Running the Application

Start the application with:

```bash
streamlit run app.py
```

The app will open in your web browser at http://localhost:8501

## Usage Guide

### 1. Upload Documents

- Click on the file uploader in the "Upload Documents" section
- Select one or more files (PDF, DOCX, TXT, CSV, HTML, MD)
- Click "Process Documents" to analyze the content

### 2. Ask Questions

- Type your question in the input field
- Click "Ask" to get an answer based on your documents
- View sources by expanding the "View Sources" section under each answer

### 3. Managing Your Database

Use the sidebar controls to:
- **Clear Files**: Remove uploaded documents
- **Clear Vector DB**: Delete the document knowledge base 
- **Reset Chat**: Clear the conversation history

## Configuration

The application settings can be customized in the `config.yaml` file:

```yaml
ollama:
  base_url: "http://localhost:11434"  # Ollama server URL
  model: "qwen3:0.6b"                 # Model name
  temperature: 0.1                    # Response creativity (0-1)

storage:
  vectorstore_path: "vectorstore.faiss"  # Vector database location
  upload_dir: "uploaded_files"           # Document storage location

document_processing:
  chunk_size: 1000                    # Size of text chunks
  chunk_overlap: 200                  # Overlap between chunks

retrieval:
  k_documents: 4                      # Number of chunks to retrieve per query
```

## Technical Details

DocumentQA uses:
- **LangChain**: For document processing and RAG implementation
- **FAISS**: For vector similarity search
- **Ollama**: For embeddings and text generation
- **Streamlit**: For the user interface

## Troubleshooting

### Common Issues

1. **"Failed to initialize AI" error**
   - Make sure Ollama is running (`ollama serve`)
   - Verify you've downloaded the required model (`ollama pull qwen3:0.6b`)

2. **"Error loading vectorstore" error**
   - The vector database may be corrupted; click "Clear Vector DB" in the sidebar

3. **Slow performance**
   - Try a smaller model in `config.yaml` (e.g., "llama3:1b")
   - Reduce `chunk_size` for faster processing of large documents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Ollama](https://ollama.com/) for local model hosting