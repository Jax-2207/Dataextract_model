"""
Configuration settings for Multimodal RAG application
Supports both local and cloud-based processing
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FAISS_DIR = os.path.join(DATA_DIR, "faiss")

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, FAISS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# CLOUD API SETTINGS (FREE TIER)
# =============================================================================

# Groq API - Free: 14,400 requests/day, ultra-fast inference
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"  # Best free model
GROQ_WHISPER_MODEL = "whisper-large-v3"  # Much more accurate than local tiny

# Cohere API - Free: 1000 requests/min
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_EMBED_MODEL = "embed-english-v3.0"  # 1024 dimensions, state-of-the-art

# Cloudinary - Free: 25GB storage
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")

# Toggle cloud vs local processing
USE_CLOUD_LLM = os.getenv("USE_CLOUD_LLM", "true").lower() == "true" and GROQ_API_KEY
USE_CLOUD_WHISPER = os.getenv("USE_CLOUD_WHISPER", "true").lower() == "true" and GROQ_API_KEY
USE_CLOUD_EMBEDDINGS = os.getenv("USE_CLOUD_EMBEDDINGS", "true").lower() == "true" and COHERE_API_KEY
USE_CLOUDINARY = os.getenv("USE_CLOUDINARY", "false").lower() == "true" and CLOUDINARY_CLOUD_NAME

# =============================================================================
# LOCAL FALLBACK SETTINGS
# =============================================================================

# Embedding settings - used when cloud is unavailable
EMBEDDING_MODEL = "BAAI/bge-base-en"  # 768 dimensions
LOCAL_EMBEDDING_DIM = 768

# Dynamic embedding dimension based on provider
EMBEDDING_DIM = 1024 if USE_CLOUD_EMBEDDINGS else LOCAL_EMBEDDING_DIM

# Chunking settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# Local LLM settings (Ollama fallback)
LLM_MODEL_ID = "meta-llama/Llama-3-8B-Instruct"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Local Whisper settings (fallback)
WHISPER_MODEL = "tiny"

# =============================================================================
# OCR SETTINGS
# =============================================================================
USE_PADDLEOCR = True
PADDLEOCR_LANG = "en"

# =============================================================================
# DATABASE SETTINGS
# =============================================================================
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "multimodal_rag")

# FAISS settings
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
FAISS_MAPPING_PATH = os.path.join(FAISS_DIR, "mapping.json")

# Search settings
TOP_K_RESULTS = 10

# =============================================================================
# STARTUP INFO
# =============================================================================
def print_config_status():
    """Print current configuration status"""
    print("\n" + "="*60)
    print("DataExtract RAG - Configuration Status")
    print("="*60)
    print(f"  LLM:          {'☁️ Groq (llama-3.3-70b)' if USE_CLOUD_LLM else '💻 Local Ollama'}")
    print(f"  Whisper:      {'☁️ Groq (whisper-large-v3)' if USE_CLOUD_WHISPER else '💻 Local Whisper'}")
    print(f"  Embeddings:   {'☁️ Cohere (embed-v3)' if USE_CLOUD_EMBEDDINGS else '💻 Local BGE'}")
    print(f"  File Storage: {'☁️ Cloudinary' if USE_CLOUDINARY else '💻 Local Storage'}")
    print("="*60 + "\n")
