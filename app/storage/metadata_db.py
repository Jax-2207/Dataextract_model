"""
PostgreSQL Metadata Database for storing document information
"""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# Try to import psycopg2, but make it optional
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from app.config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD
)


def get_connection():
    """Get a PostgreSQL connection"""
    if not POSTGRES_AVAILABLE:
        raise RuntimeError("psycopg2 not installed. Run: pip install psycopg2-binary")
    
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )


def init_database():
    """Initialize the database tables"""
    if not POSTGRES_AVAILABLE:
        print("Warning: PostgreSQL not available. Skipping database initialization.")
        return
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                file_size BIGINT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                chunk_count INTEGER DEFAULT 0,
                metadata JSONB
            )
        """)
        
        # Create chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text_content TEXT NOT NULL,
                vector_index INTEGER,
                metadata JSONB
            )
        """)
        
        # Create processing_logs table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                status VARCHAR(50) NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("Database tables initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")


def insert_document(
    filename: str,
    file_path: str,
    file_type: str,
    file_size: Optional[int] = None,
    metadata: Optional[Dict] = None
) -> Optional[int]:
    """Insert a new document record"""
    if not POSTGRES_AVAILABLE:
        return None
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO documents (filename, file_path, file_type, file_size, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (filename, file_path, file_type, file_size, 
              psycopg2.extras.Json(metadata) if metadata else None))
        
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return doc_id
        
    except Exception as e:
        print(f"Error inserting document: {e}")
        return None


def update_document_processed(doc_id: int, chunk_count: int):
    """Mark document as processed with chunk count"""
    if not POSTGRES_AVAILABLE:
        return
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            UPDATE documents 
            SET processed = TRUE, chunk_count = %s
            WHERE id = %s
        """, (chunk_count, doc_id))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error updating document: {e}")


def insert_chunks(doc_id: int, chunks: List[Dict[str, Any]]):
    """Insert multiple chunks for a document"""
    if not POSTGRES_AVAILABLE:
        return
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        for chunk in chunks:
            cur.execute("""
                INSERT INTO chunks (document_id, chunk_index, text_content, vector_index, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                doc_id,
                chunk.get('chunk_id', 0),
                chunk.get('text', ''),
                chunk.get('vector_index'),
                psycopg2.extras.Json(chunk) if chunk else None
            ))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error inserting chunks: {e}")


def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    """Get document by ID"""
    if not POSTGRES_AVAILABLE:
        return None
    
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        result = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return dict(result) if result else None
        
    except Exception as e:
        print(f"Error getting document: {e}")
        return None


def get_all_documents() -> List[Dict[str, Any]]:
    """Get all documents"""
    if not POSTGRES_AVAILABLE:
        return []
    
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("SELECT * FROM documents ORDER BY upload_time DESC")
        results = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [dict(r) for r in results]
        
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []


def log_processing(doc_id: int, status: str, message: Optional[str] = None):
    """Log a processing event"""
    if not POSTGRES_AVAILABLE:
        return
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO processing_logs (document_id, status, message)
            VALUES (%s, %s, %s)
        """, (doc_id, status, message))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error logging processing: {e}")


def delete_document(doc_id: int) -> bool:
    """Delete a document and its chunks"""
    if not POSTGRES_AVAILABLE:
        return False
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
