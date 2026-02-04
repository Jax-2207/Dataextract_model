"""
Text chunking utilities for document processing
"""
from typing import List, Dict, Any


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The input text to chunk
        size: Maximum size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    
    return chunks


def chunk_text_by_sentences(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks by sentences, respecting max chunk size.
    Better for maintaining semantic coherence.
    
    Args:
        text: The input text to chunk
        max_chunk_size: Maximum size of each chunk
    
    Returns:
        List of text chunks
    """
    import re
    
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_with_metadata(
    text: str, 
    file_path: str, 
    chunk_size: int = 500, 
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Chunk text and attach metadata to each chunk.
    
    Args:
        text: The input text to chunk
        file_path: Source file path
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of dictionaries with chunk text and metadata
    """
    import os
    
    chunks = chunk_text(text, chunk_size, overlap)
    
    chunks_with_metadata = []
    for idx, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "text": chunk,
            "chunk_id": idx,
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "start_char": idx * (chunk_size - overlap),
            "end_char": idx * (chunk_size - overlap) + len(chunk)
        })
    
    return chunks_with_metadata
