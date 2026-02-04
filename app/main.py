"""
FastAPI entry point for Multimodal RAG application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, query

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

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.get("/")
def home():
    """Root endpoint - health check"""
    return {
        "message": "Multimodal RAG backend is running",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/stats")
def get_stats():
    """Get system statistics"""
    try:
        from app.storage.vector_store import get_stats
        vector_stats = get_stats()
    except:
        vector_stats = {"error": "Could not get vector store stats"}
    
    return {
        "vector_store": vector_stats
    }