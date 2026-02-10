"""
Smart RAG Query API endpoint with Confidence Scoring and Self-Learning
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.core.embeddings import get_query_embeddings
from app.storage.vector_store import search_vectors, get_chunks_by_indices
from app.core.llm import generate_with_confidence, generate_internet_answer
from app.storage.learned_answers import (
    search_learned_answer, 
    save_learned_answer,
    get_learned_stats
)
from app.utils.diversity import ensure_file_diversity

router = APIRouter()

# Confidence thresholds
LOCAL_DB_THRESHOLD = 60  # Below this, offer internet option
LEARNING_THRESHOLD = 90  # Above this, save internet answer to DB


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class SmartQueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: int
    source: str  # "local_db", "learned", "internet"
    sources: List[dict]
    offer_internet: bool
    reasoning: Optional[str] = None


class InternetQueryRequest(BaseModel):
    question: str
    save_if_confident: Optional[bool] = True


class InternetQueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: int
    source: str
    saved_to_db: bool
    reasoning: Optional[str] = None


@router.post("/", response_model=SmartQueryResponse)
async def smart_query(request: QueryRequest):
    """
    Smart RAG Query with confidence scoring:
    1. Check learned answers first
    2. Search local DB (uploaded files)
    3. Generate answer with confidence score
    4. If confidence < 60%, offer internet option
    """
    try:
        # Step 1: Check if we have a learned answer for this question
        learned = search_learned_answer(request.question)
        if learned:
            return SmartQueryResponse(
                question=request.question,
                answer=learned["answer"],
                confidence_score=learned["confidence_score"],
                source="learned",
                sources=[{"type": "learned_answer", "original_source": learned.get("source", "internet")}],
                offer_internet=False,
                reasoning="This answer was previously learned and saved."
            )
        
        # Step 2: Search local vector DB (retrieve more for diversity)
        question_embedding = get_query_embeddings(request.question)
        # Retrieve 2x chunks initially to ensure multi-document coverage
        search_k = min(request.top_k * 2, 40)
        distances, indices = search_vectors(question_embedding, k=search_k)
        chunks_data = get_chunks_by_indices(indices[0])
        
        # Step 3: If no local data, offer internet
        if not chunks_data:
            return SmartQueryResponse(
                question=request.question,
                answer="No documents have been uploaded yet. Would you like me to search using my general knowledge?",
                confidence_score=0,
                source="none",
                sources=[],
                offer_internet=True,
                reasoning="No documents found in the database."
            )
        
        # Step 3.5: Ensure file diversity in results
        chunks_data = ensure_file_diversity(
            chunks_data, 
            max_chunks=request.top_k,
            max_per_file=max(3, request.top_k // 2)  # Allow at least 3 per file
        )
        
        # Step 3.6: Analyze question type for intelligent answering
        from app.utils.query_understanding import get_question_context
        question_context = get_question_context(request.question)
        
        # Step 4: Build enriched context with file metadata
        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks_data, 1):
            file_name = chunk.get("file", "unknown")
            chunk_text = chunk["text"]
            
            # Add metadata header for each chunk to help LLM understand sources
            context_parts.append(f"[Source {i}: {file_name}]\n{chunk_text}")
            
            sources.append({
                "file": file_name,
                "chunk_id": chunk.get("chunk_id", -1)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer with confidence score and question understanding
        result = generate_with_confidence(
            prompt="",  # Not used in new function
            context=context,
            question=request.question,
            question_type=question_context['type'],
            guidance=question_context['guidance']
        )
        
        confidence = result["confidence_score"]
        offer_internet = confidence < LOCAL_DB_THRESHOLD
        
        return SmartQueryResponse(
            question=request.question,
            answer=result["answer"],
            confidence_score=confidence,
            source="local_db",
            sources=sources,
            offer_internet=offer_internet,
            reasoning=result.get("reasoning", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/internet", response_model=InternetQueryResponse)
async def internet_query(request: InternetQueryRequest):
    """
    Search using LLM's general knowledge (internet mode).
    If confidence >= 90%, save to learned answers DB for future use.
    """
    try:
        # Generate answer using internet/general knowledge
        result = generate_internet_answer(request.question)
        
        confidence = result["confidence_score"]
        saved = False
        
        # Save to learned answers if confidence is high enough
        if request.save_if_confident and confidence >= LEARNING_THRESHOLD:
            saved = save_learned_answer(
                question=request.question,
                answer=result["answer"],
                confidence_score=confidence
            )
        
        return InternetQueryResponse(
            question=request.question,
            answer=result["answer"],
            confidence_score=confidence,
            source="internet",
            saved_to_db=saved,
            reasoning=result.get("reasoning", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with internet query: {str(e)}")


@router.get("/learned-stats")
async def get_stats():
    """Get statistics about learned answers"""
    stats = get_learned_stats()
    return {
        "total_learned_answers": stats["total"],
        "average_confidence": stats["avg_confidence"],
        "learning_threshold": LEARNING_THRESHOLD,
        "local_db_threshold": LOCAL_DB_THRESHOLD
    }


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "smart_query"}
