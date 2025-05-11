import os
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default config values
DEFAULT_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "qwen3:0.6b",
        "temperature": 0.1,
        "context_window": 2048,
        "timeout": 120
    },
    "storage": {
        "vectorstore_path": "vectorstore.faiss",
        "upload_dir": "uploaded_files",
        "log_dir": "logs",
        "backup_dir": "backups"
    },
    "document_processing": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "supported_extensions": [".pdf", ".docx", ".txt", ".csv", ".pptx", ".md", ".html"]
    },
    "retrieval": {
        "k_documents": 4,
        "similarity_threshold": 0.7
    },
    "ui": {
        "theme": "light",
        "max_history_items": 50,
        "default_page_title": "Document Q&A"
    }
}

# Try to load config from file, fall back to defaults
CONFIG_FILE = "config.yaml"


def load_config():
    """
    Load configuration from YAML file or create with defaults if not exists.

    Returns:
        Dict containing configuration values
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {CONFIG_FILE}")
                return config
        else:
            # Create default config file
            with open(CONFIG_FILE, 'w') as file:
                yaml.dump(DEFAULT_CONFIG, file, default_flow_style=False)
                logger.info(f"Default configuration created at {CONFIG_FILE}")
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return DEFAULT_CONFIG


# Load config once at module import
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
for directory in [UPLOAD_DIR, config["storage"]["log_dir"], config["storage"]["backup_dir"]]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def update_config(section, key, value):
    """
    Update a configuration value and save to file.

    Args:
        section: Config section (e.g., 'ollama', 'storage')
        key: Config key within section
        value: New value to set

    Returns:
        Boolean indicating success
    """
    try:
        if section not in config:
            logger.error(f"Config section '{section}' not found")
            return False

        if key not in config[section]:
            logger.error(f"Config key '{key}' not found in section '{section}'")
            return False

        # Update config in memory
        config[section][key] = value

        # Update module-level variable if it exists
        var_name = f"{section.upper()}_{key.upper()}"
        if var_name in globals():
            globals()[var_name] = value

        # Save to file
        with open(CONFIG_FILE, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        logger.info(f"Updated config: {section}.{key} = {value}")
        return True

    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return False


def get_config_summary():
    """
    Get a summary of current configuration.

    Returns:
        Dict with simplified config for display
    """
    return {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "vectorstore": VECTORSTORE_PATH,
        "upload_dir": UPLOAD_DIR,
        "chunk_size": CHUNK_SIZE,
        "document_count": count_documents()
    }


def count_documents():
    """
    Count documents in upload directory.

    Returns:
        Number of documents in upload directory
    """
    try:
        if not os.path.exists(UPLOAD_DIR):
            return 0

        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        return len(files)
    except Exception:
        return 0