"""
File Upload API endpoint
"""
import os
import shutil
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.router import detect_file_type
from app.core.ingestion import ingest_and_process, save_uploaded_file, extract_content
from app.config import UPLOAD_DIR

router = APIRouter()


class UploadResponse(BaseModel):
    status: str
    file_name: str
    file_type: str
    text_preview: str
    chunks_created: int = 0
    message: str = ""


@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a file.
    
    Supported formats:
    - PDF: .pdf
    - Images: .jpg, .jpeg, .png
    - Audio: .mp3, .wav
    - Video: .mp4
    
    Returns:
        Processing status and text preview
    """
    try:
        # Detect file type
        file_type = detect_file_type(file.filename)
        
        if file_type == "unknown":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )
        
        # Process file through ingestion pipeline
        result = ingest_and_process(file)
        
        # Get text preview (first 500 chars)
        text_preview = ""
        if result.get("status") == "success":
            from app.storage.vector_store import get_chunks_by_indices
            
            indices = result.get("vector_indices", [])
            if indices:
                chunks = get_chunks_by_indices(indices[:1])  # Get first chunk
                if chunks:
                    text_preview = chunks[0].get("text", "")[:500]
        
        return UploadResponse(
            status=result.get("status", "unknown"),
            file_name=file.filename,
            file_type=file_type,
            text_preview=text_preview,
            chunks_created=result.get("chunks_created", 0),
            message=result.get("error", "File processed successfully")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/simple")
async def upload_file_simple(file: UploadFile = File(...)):
    """
    Simple upload endpoint - just extract text without storing in vector DB.
    Good for testing OCR and extraction.
    """
    try:
        # Save file temporarily
        file_path = save_uploaded_file(file)
        file_type = detect_file_type(file.filename)
        
        # Extract text
        text = extract_content(file_path, file_type)
        
        return {
            "file_name": file.filename,
            "file_type": file_type,
            "text_length": len(text),
            "text_preview": text[:1000] if text else "",
            "full_text": text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/batch")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple files at once.
    """
    results = []
    
    for file in files:
        try:
            file_type = detect_file_type(file.filename)
            
            if file_type == "unknown":
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "message": "Unsupported file type"
                })
                continue
            
            # Reset file position for each file
            await file.seek(0)
            
            result = ingest_and_process(file)
            results.append({
                "file_name": file.filename,
                "file_type": file_type,
                "status": result.get("status"),
                "chunks_created": result.get("chunks_created", 0)
            })
            
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {
        "total_files": len(files),
        "results": results
    }


@router.get("/supported-types")
def get_supported_types():
    """Get list of supported file types"""
    return {
        "pdf": [".pdf"],
        "image": [".jpg", ".jpeg", ".png"],
        "audio": [".mp3", ".wav"],
        "video": [".mp4"]
    }
