"""
FastAPI entry point for Multimodal RAG application
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import upload, query

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG",
    description="Multimodal Retrieval-Augmented Generation API supporting PDF, Audio, Video, and Images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.get("/")
def home():
    """Serve the main UI"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Multimodal RAG backend is running",
        "version": "1.0.0",
        "ui": "Static files not found. Place index.html in /static folder.",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Comprehensive health check endpoint for dashboard"""
    health = {
        "status": "healthy",
        "components": {}
    }
    
    # Check MongoDB
    try:
        from app.storage.metadata_db import test_connection
        if test_connection():
            health["components"]["mongodb"] = {"status": "healthy", "message": "Connected"}
        else:
            health["components"]["mongodb"] = {"status": "unhealthy", "message": "Connection failed"}
    except Exception as e:
        health["components"]["mongodb"] = {"status": "unhealthy", "message": str(e)}
    
    # Check FAISS
    try:
        from app.storage.vector_store import get_stats
        stats = get_stats()
        health["components"]["faiss"] = {
            "status": "healthy",
            "vectors": stats.get("total_vectors", 0),
            "embedding_dim": stats.get("embedding_dim", 0)
        }
    except Exception as e:
        health["components"]["faiss"] = {"status": "unhealthy", "message": str(e)}
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            health["components"]["ollama"] = {
                "status": "healthy",
                "models": models
            }
        else:
            health["components"]["ollama"] = {"status": "unhealthy", "message": "Not responding"}
    except:
        health["components"]["ollama"] = {"status": "unhealthy", "message": "Offline"}
    
    return health


@app.get("/stats")
def get_stats():
    """Get detailed system statistics for dashboard"""
    stats = {
        "vector_store": {},
        "database": {},
        "processors": {}
    }
    
    # Vector store stats
    try:
        from app.storage.vector_store import get_stats as get_vector_stats
        stats["vector_store"] = get_vector_stats()
    except Exception as e:
        stats["vector_store"] = {"error": str(e)}
    
    # Database stats
    try:
        from app.storage.metadata_db import get_all_documents
        docs = get_all_documents()
        stats["database"]["document_count"] = len(docs)
    except:
        stats["database"]["document_count"] = 0
    
    # Processor availability
    processors = {
        "pdf": "PyMuPDF",
        "docx": "python-docx",
        "xlsx": "openpyxl",
        "pptx": "python-pptx",
        "image": "pytesseract",
        "audio": "whisper",
        "video": "ffmpeg+whisper"
    }
    
    for proc_type, proc_name in processors.items():
        try:
            if proc_type == "pdf":
                import fitz
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "docx":
                import docx
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "xlsx":
                import openpyxl
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "pptx":
                from pptx import Presentation
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "image":
                import pytesseract
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "audio":
                import whisper
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
            elif proc_type == "video":
                stats["processors"][proc_type] = {"status": "available", "name": proc_name}
        except:
            stats["processors"][proc_type] = {"status": "unavailable", "name": proc_name}
    
    return stats