"""
FAISS Vector Store for storing and retrieving embeddings
"""
import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import faiss

from app.config import EMBEDDING_DIM, FAISS_INDEX_PATH, FAISS_MAPPING_PATH, FAISS_DIR

# Global instances
_index: Optional[faiss.Index] = None
_chunk_mapping: Dict[int, Dict[str, Any]] = {}


def get_index() -> faiss.Index:
    """Get or create the FAISS index"""
    global _index
    
    if _index is None:
        if os.path.exists(FAISS_INDEX_PATH):
            _index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            _index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    return _index


def get_chunk_mapping() -> Dict[int, Dict[str, Any]]:
    """Get or load the chunk mapping"""
    global _chunk_mapping
    
    if not _chunk_mapping and os.path.exists(FAISS_MAPPING_PATH):
        with open(FAISS_MAPPING_PATH, 'r') as f:
            # Convert string keys back to integers
            loaded = json.load(f)
            _chunk_mapping = {int(k): v for k, v in loaded.items()}
    
    return _chunk_mapping


def save_index():
    """Save the FAISS index to disk"""
    os.makedirs(FAISS_DIR, exist_ok=True)
    index = get_index()
    faiss.write_index(index, FAISS_INDEX_PATH)


def save_mapping():
    """Save the chunk mapping to disk"""
    os.makedirs(FAISS_DIR, exist_ok=True)
    mapping = get_chunk_mapping()
    with open(FAISS_MAPPING_PATH, 'w') as f:
        json.dump(mapping, f)


def add_embeddings(embeddings: np.ndarray, chunks_data: List[Dict[str, Any]]) -> List[int]:
    """
    Add embeddings to the vector store.
    
    Args:
        embeddings: numpy array of embeddings (n_samples, embedding_dim)
        chunks_data: List of chunk metadata dictionaries
    
    Returns:
        List of indices where embeddings were added
    """
    index = get_index()
    mapping = get_chunk_mapping()
    
    # Ensure embeddings are float32
    embeddings = embeddings.astype('float32')
    
    # Get current size for new indices
    start_idx = index.ntotal
    
    # Add to FAISS index
    index.add(embeddings)
    
    # Store metadata in mapping
    indices = []
    for i, chunk_data in enumerate(chunks_data):
        idx = start_idx + i
        # Remove embedding from metadata to save space (it's in FAISS)
        chunk_copy = {k: v for k, v in chunk_data.items() if k != 'embedding'}
        mapping[idx] = chunk_copy
        indices.append(idx)
    
    # Save to disk
    save_index()
    save_mapping()
    
    return indices


def search_vectors(
    query_embedding: np.ndarray, 
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors.
    
    Args:
        query_embedding: The query embedding (1, embedding_dim) or (embedding_dim,)
        k: Number of results to return
    
    Returns:
        Tuple of (distances, indices)
    """
    index = get_index()
    
    # Ensure proper shape
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype('float32')
    
    # Limit k to available vectors
    k = min(k, index.ntotal)
    
    if k == 0:
        return np.array([[]]), np.array([[]])
    
    distances, indices = index.search(query_embedding, k)
    
    return distances, indices


def get_chunks_by_indices(indices: np.ndarray) -> List[Dict[str, Any]]:
    """
    Get chunk metadata by their indices.
    
    Args:
        indices: Array of indices to retrieve
    
    Returns:
        List of chunk metadata dictionaries
    """
    mapping = get_chunk_mapping()
    chunks = []
    
    for idx in indices:
        idx = int(idx)
        if idx in mapping:
            chunks.append(mapping[idx])
    
    return chunks


def delete_by_file(file_path: str) -> int:
    """
    Delete all chunks associated with a file.
    Note: FAISS IndexFlatL2 doesn't support deletion, 
    so we mark them in mapping and rebuild periodically.
    
    Args:
        file_path: Path of the file to remove
    
    Returns:
        Number of chunks marked for deletion
    """
    mapping = get_chunk_mapping()
    count = 0
    
    for idx, data in list(mapping.items()):
        if data.get('file_path') == file_path:
            mapping[idx]['deleted'] = True
            count += 1
    
    save_mapping()
    return count


def get_stats() -> Dict[str, Any]:
    """Get vector store statistics"""
    index = get_index()
    mapping = get_chunk_mapping()
    
    return {
        "total_vectors": index.ntotal,
        "total_chunks": len(mapping),
        "embedding_dim": EMBEDDING_DIM,
        "index_trained": index.is_trained
    }


def clear_all():
    """Clear all data from the vector store"""
    global _index, _chunk_mapping
    
    _index = faiss.IndexFlatL2(EMBEDDING_DIM)
    _chunk_mapping = {}
    
    save_index()
    save_mapping()
