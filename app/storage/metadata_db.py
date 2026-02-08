"""
MongoDB Atlas Metadata Database for storing document information
"""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# Try to import pymongo, but make it optional
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

from app.config import MONGODB_URI, MONGODB_DB_NAME


# Global client instance for connection reuse
_client: Optional["MongoClient"] = None
_db = None


def get_client():
    """Get or create a MongoDB client (singleton pattern)"""
    global _client
    if not MONGODB_AVAILABLE:
        raise RuntimeError("pymongo not installed. Run: pip install pymongo dnspython")
    
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client


def get_database():
    """Get the database instance"""
    global _db
    if _db is None:
        client = get_client()
        _db = client[MONGODB_DB_NAME]
    return _db


def init_database():
    """Initialize the database collections and indexes"""
    if not MONGODB_AVAILABLE:
        print("Warning: MongoDB not available. Skipping database initialization.")
        return
    
    try:
        db = get_database()
        
        # Create indexes for better query performance
        # Documents collection
        db.documents.create_index("filename")
        db.documents.create_index("file_type")
        db.documents.create_index("upload_time")
        db.documents.create_index("processed")
        
        # Chunks collection
        db.chunks.create_index("document_id")
        db.chunks.create_index([("document_id", 1), ("chunk_index", 1)])
        
        # Processing logs collection
        db.processing_logs.create_index("document_id")
        db.processing_logs.create_index("timestamp")
        
        print("MongoDB collections and indexes initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")


def insert_document(
    filename: str,
    file_path: str,
    file_type: str,
    file_size: Optional[int] = None,
    metadata: Optional[Dict] = None
) -> Optional[str]:
    """Insert a new document record. Returns the document ID as string."""
    if not MONGODB_AVAILABLE:
        return None
    
    try:
        db = get_database()
        
        document = {
            "filename": filename,
            "file_path": file_path,
            "file_type": file_type,
            "file_size": file_size,
            "upload_time": datetime.utcnow(),
            "processed": False,
            "chunk_count": 0,
            "metadata": metadata or {}
        }
        
        result = db.documents.insert_one(document)
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error inserting document: {e}")
        return None


def update_document_processed(doc_id: str, chunk_count: int):
    """Mark document as processed with chunk count"""
    if not MONGODB_AVAILABLE:
        return
    
    try:
        db = get_database()
        
        db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"processed": True, "chunk_count": chunk_count}}
        )
        
    except Exception as e:
        print(f"Error updating document: {e}")


def insert_chunks(doc_id: str, chunks: List[Dict[str, Any]]):
    """Insert multiple chunks for a document"""
    if not MONGODB_AVAILABLE:
        return
    
    try:
        db = get_database()
        
        chunk_documents = []
        for chunk in chunks:
            chunk_doc = {
                "document_id": doc_id,
                "chunk_index": chunk.get("chunk_id", 0),
                "text_content": chunk.get("text", ""),
                "vector_index": chunk.get("vector_index"),
                "metadata": chunk
            }
            chunk_documents.append(chunk_doc)
        
        if chunk_documents:
            db.chunks.insert_many(chunk_documents)
        
    except Exception as e:
        print(f"Error inserting chunks: {e}")


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Get document by ID"""
    if not MONGODB_AVAILABLE:
        return None
    
    try:
        db = get_database()
        
        result = db.documents.find_one({"_id": ObjectId(doc_id)})
        
        if result:
            result["id"] = str(result.pop("_id"))
            return result
        return None
        
    except Exception as e:
        print(f"Error getting document: {e}")
        return None


def get_all_documents() -> List[Dict[str, Any]]:
    """Get all documents"""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        db = get_database()
        
        results = db.documents.find().sort("upload_time", -1)
        
        documents = []
        for doc in results:
            doc["id"] = str(doc.pop("_id"))
            documents.append(doc)
        
        return documents
        
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []


def get_document_chunks(doc_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a document"""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        db = get_database()
        
        results = db.chunks.find({"document_id": doc_id}).sort("chunk_index", 1)
        
        chunks = []
        for chunk in results:
            chunk["id"] = str(chunk.pop("_id"))
            chunks.append(chunk)
        
        return chunks
        
    except Exception as e:
        print(f"Error getting chunks: {e}")
        return []


def log_processing(doc_id: str, status: str, message: Optional[str] = None):
    """Log a processing event"""
    if not MONGODB_AVAILABLE:
        return
    
    try:
        db = get_database()
        
        log_entry = {
            "document_id": doc_id,
            "status": status,
            "message": message,
            "timestamp": datetime.utcnow()
        }
        
        db.processing_logs.insert_one(log_entry)
        
    except Exception as e:
        print(f"Error logging processing: {e}")


def get_processing_logs(doc_id: str) -> List[Dict[str, Any]]:
    """Get processing logs for a document"""
    if not MONGODB_AVAILABLE:
        return []
    
    try:
        db = get_database()
        
        results = db.processing_logs.find({"document_id": doc_id}).sort("timestamp", 1)
        
        logs = []
        for log in results:
            log["id"] = str(log.pop("_id"))
            logs.append(log)
        
        return logs
        
    except Exception as e:
        print(f"Error getting processing logs: {e}")
        return []


def delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks"""
    if not MONGODB_AVAILABLE:
        return False
    
    try:
        db = get_database()
        
        # Delete associated chunks first
        db.chunks.delete_many({"document_id": doc_id})
        
        # Delete processing logs
        db.processing_logs.delete_many({"document_id": doc_id})
        
        # Delete the document
        result = db.documents.delete_one({"_id": ObjectId(doc_id)})
        
        return result.deleted_count > 0
        
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False


def close_connection():
    """Close the MongoDB connection"""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


def test_connection() -> bool:
    """Test the MongoDB connection"""
    if not MONGODB_AVAILABLE:
        return False
    
    try:
        client = get_client()
        # The ping command is cheap and doesn't require auth
        client.admin.command('ping')
        print("MongoDB connection successful!")
        return True
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        return False
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False
