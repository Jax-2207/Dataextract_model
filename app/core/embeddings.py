"""
Embedding generation using sentence-transformers
"""
from typing import List, Union
import numpy as np

# Global model instance (lazy loading)
_model = None


def get_model():
    """Lazy load the embedding model"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate embeddings for text(s).
    
    Args:
        texts: Single string or list of strings to embed
    
    Returns:
        numpy array of embeddings with shape (n_texts, embedding_dim)
    """
    model = get_model()
    
    if isinstance(texts, str):
        texts = [texts]
    
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Ensure 2D array
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    
    return embeddings.astype('float32')


def embed_chunks(chunks: List[dict]) -> List[dict]:
    """
    Add embeddings to chunks with metadata.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
    
    Returns:
        Same chunks with 'embedding' field added
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0 to 1)
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
