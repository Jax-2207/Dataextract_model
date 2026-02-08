"""
Ingestion pipeline for processing uploaded files
"""
import os
import shutil
from typing import Dict, Any, Optional, Tuple

from app.core.router import detect_file_type
from app.core.chunking import chunk_with_metadata
from app.core.embeddings import get_embeddings
from app.storage.vector_store import add_embeddings
from app.config import UPLOAD_DIR, PROCESSED_DIR


def save_uploaded_file(file, upload_dir: str = UPLOAD_DIR) -> str:
    """
    Save uploaded file to disk.
    
    Args:
        file: FastAPI UploadFile object
        upload_dir: Directory to save file
    
    Returns:
        Path to saved file
    """
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_path


def ingest_file(file) -> Tuple[str, str, str]:
    """
    Process an uploaded file through the ingestion pipeline.
    
    Args:
        file: FastAPI UploadFile object
    
    Returns:
        Tuple of (file_path, file_type, extracted_text)
    """
    # Save file
    file_path = save_uploaded_file(file)
    
    # Detect type
    file_type = detect_file_type(file.filename)
    
    # Extract text based on file type
    extracted_text = extract_content(file_path, file_type)
    
    return file_path, file_type, extracted_text


def extract_content(file_path: str, file_type: str) -> str:
    """
    Extract text content based on file type.
    
    Args:
        file_path: Path to file
        file_type: Detected file type
    
    Returns:
        Extracted text content
    """
    if file_type == "pdf":
        from app.processors.pdf import extract_text_from_pdf
        return extract_text_from_pdf(file_path)
    
    elif file_type == "image":
        from app.processors.image import extract_text_from_image
        return extract_text_from_image(file_path)
    
    elif file_type == "audio":
        from app.processors.audio import audio_to_text
        return audio_to_text(file_path)
    
    elif file_type == "video":
        from app.processors.video import video_to_text
        return video_to_text(file_path)
    
    elif file_type == "docx":
        from app.processors.document import extract_text_from_docx
        return extract_text_from_docx(file_path)
    
    elif file_type == "xlsx":
        from app.processors.document import extract_text_from_xlsx
        return extract_text_from_xlsx(file_path)
    
    elif file_type == "pptx":
        from app.processors.document import extract_text_from_pptx
        return extract_text_from_pptx(file_path)
    
    else:
        return ""


def process_and_store(
    file_path: str, 
    file_type: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, Any]:
    """
    Full ingestion pipeline: extract, chunk, embed, and store.
    
    Args:
        file_path: Path to file
        file_type: Detected file type
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        Dictionary with processing results
    """
    result = {
        "file_path": file_path,
        "file_type": file_type,
        "status": "processing"
    }
    
    try:
        # Step 1: Extract text
        text = extract_content(file_path, file_type)
        result["text_length"] = len(text)
        
        if not text.strip():
            result["status"] = "no_text_extracted"
            result["chunks_created"] = 0
            return result
        
        # Step 2: Clean text
        from app.utils.text_cleaner import clean_text
        text = clean_text(text)
        
        # Step 3: Chunk text
        chunks = chunk_with_metadata(
            text, 
            file_path, 
            chunk_size=chunk_size, 
            overlap=chunk_overlap
        )
        result["chunks_created"] = len(chunks)
        
        if not chunks:
            result["status"] = "no_chunks_created"
            return result
        
        # Step 4: Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)
        
        # Step 5: Store in vector store
        indices = add_embeddings(embeddings, chunks)
        result["vector_indices"] = indices
        
        result["status"] = "success"
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def ingest_and_process(file) -> Dict[str, Any]:
    """
    Complete ingestion workflow for an uploaded file.
    
    Args:
        file: FastAPI UploadFile object
    
    Returns:
        Dictionary with full processing results
    """
    # Save and detect type
    file_path, file_type, _ = ingest_file(file)
    
    # Process and store
    result = process_and_store(file_path, file_type)
    
    return result


def batch_ingest(files: list) -> list:
    """
    Ingest multiple files.
    
    Args:
        files: List of FastAPI UploadFile objects
    
    Returns:
        List of processing results
    """
    results = []
    
    for file in files:
        result = ingest_and_process(file)
        results.append(result)
    
    return results
