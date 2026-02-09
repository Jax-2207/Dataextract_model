"""
Embedding generation with Cohere API (cloud) and sentence-transformers fallback
"""
from typing import List, Union
import numpy as np

from app.config import USE_CLOUD_EMBEDDINGS, COHERE_API_KEY, COHERE_EMBED_MODEL, EMBEDDING_DIM

# Global model instance (lazy loading for local fallback)
_local_model = None
_cohere_client = None


def get_cohere_client():
    """Get or create Cohere client"""
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(api_key=COHERE_API_KEY)
    return _cohere_client


def get_local_model():
    """Lazy load the local embedding model"""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import EMBEDDING_MODEL
        print(f"Loading local embedding model '{EMBEDDING_MODEL}'...")
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Local embedding model loaded!")
    return _local_model


def get_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate embeddings for text(s).
    Uses Cohere API (cloud) or local sentence-transformers based on config.
    
    Args:
        texts: Single string or list of strings to embed
    
    Returns:
        numpy array of embeddings with shape (n_texts, embedding_dim)
    """
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype='float32')
    
    if USE_CLOUD_EMBEDDINGS:
        return get_cohere_embeddings(texts)
    else:
        return get_local_embeddings(texts)


def get_cohere_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using Cohere API (FREE, state-of-the-art).
    
    Uses embed-english-v3.0 - 1024 dimensions, excellent quality.
    Free tier: 1000 requests/min
    """
    try:
        client = get_cohere_client()
        
        # Cohere has a batch limit, process in chunks of 96
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Clean texts - Cohere doesn't like empty strings
            cleaned_batch = [t if t.strip() else " " for t in batch]
            
            response = client.embed(
                texts=cleaned_batch,
                model=COHERE_EMBED_MODEL,
                input_type="search_document"  # Use "search_query" for queries
            )
            
            all_embeddings.extend(response.embeddings)
        
        embeddings = np.array(all_embeddings, dtype='float32')
        return embeddings
        
    except Exception as e:
        error_msg = str(e)
        print(f"Cohere embedding error: {error_msg}")
        
        # Fallback to local on error
        if "rate_limit" in error_msg.lower():
            print("Cohere rate limit hit, falling back to local embeddings...")
        else:
            print("Falling back to local embeddings...")
        
        return get_local_embeddings(texts)


def get_query_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """
    Generate embeddings optimized for queries (for semantic search).
    
    Args:
        texts: Query text(s)
    
    Returns:
        numpy array of embeddings
    """
    if isinstance(texts, str):
        texts = [texts]
    
    if USE_CLOUD_EMBEDDINGS:
        try:
            client = get_cohere_client()
            
            cleaned_texts = [t if t.strip() else " " for t in texts]
            
            response = client.embed(
                texts=cleaned_texts,
                model=COHERE_EMBED_MODEL,
                input_type="search_query"  # Optimized for queries
            )
            
            return np.array(response.embeddings, dtype='float32')
            
        except Exception as e:
            print(f"Cohere query embedding error: {e}, falling back to local...")
            return get_local_embeddings(texts)
    else:
        return get_local_embeddings(texts)


def get_local_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using local sentence-transformers model.
    """
    model = get_local_model()
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
