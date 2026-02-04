"""
Configuration settings for Multimodal RAG application
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, FAISS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Embedding settings
# Options: "BAAI/bge-base-en" (768), "BAAI/bge-large-en-v1.5" (1024), "intfloat/e5-large-v2" (1024)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# Chunking settings - smaller chunks with more overlap for better precision
CHUNK_SIZE = 400  # Smaller chunks capture more precise context
CHUNK_OVERLAP = 100  # More overlap prevents information loss at boundaries

# LLM settings
LLM_MODEL_ID = "meta-llama/Llama-3-8B-Instruct"
OLLAMA_MODEL = "llama3.1"  # For local Ollama usage
OLLAMA_BASE_URL = "http://localhost:11434"

# OCR settings
USE_PADDLEOCR = True  # Set to False to use Tesseract as primary
PADDLEOCR_LANG = "en"

# Whisper settings - larger model for better accuracy
# Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
WHISPER_MODEL = "large-v3"  # Best accuracy, requires ~10GB VRAM (use "medium" for less GPU)

# PostgreSQL settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "multimodal_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# FAISS settings
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
FAISS_MAPPING_PATH = os.path.join(FAISS_DIR, "mapping.json")

# Search settings - retrieve more chunks for comprehensive answers
TOP_K_RESULTS = 10  # Increased from 5 for better context coverage
