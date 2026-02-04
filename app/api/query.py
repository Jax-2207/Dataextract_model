"""
RAG Query API endpoint
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.core.embeddings import get_embeddings
from app.storage.vector_store import search_vectors, get_chunks_by_indices
from app.core.llm import generate_response

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]


@router.post("/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a RAG query:
    1. Embed the question
    2. Search vector store for relevant chunks
    3. Build context from retrieved chunks
    4. Generate response using LLM
    """
    try:
        # Step 1: Embed the question
        question_embedding = get_embeddings([request.question])
        
        # Step 2: Search for relevant chunks
        distances, indices = search_vectors(question_embedding, k=request.top_k)
        
        # Step 3: Get the actual text chunks
        chunks_data = get_chunks_by_indices(indices[0])
        
        if not chunks_data:
            return QueryResponse(
                question=request.question,
                answer="I don't have enough context to answer this question. Please upload relevant documents first.",
                sources=[]
            )
        
        # Step 4: Build context from retrieved chunks
        context_parts = []
        sources = []
        for chunk in chunks_data:
            context_parts.append(chunk["text"])
            sources.append({
                "file": chunk.get("file", "unknown"),
                "chunk_id": chunk.get("chunk_id", -1)
            })
        
        context = "\n\n".join(context_parts)
        
        # Step 5: Generate response using LLM with improved prompt
        prompt = f"""You are a knowledgeable and precise assistant. Your task is to answer the question based ONLY on the provided context.

INSTRUCTIONS:
1. Read the context carefully and extract relevant information
2. Provide a detailed, accurate answer based on the context
3. If the context contains partial information, share what you can find
4. If the answer is NOT in the context, say "Based on the provided documents, I don't have enough information to answer this question."
5. When possible, cite which part of the context supports your answer
6. Be specific and avoid vague generalizations

CONTEXT FROM UPLOADED DOCUMENTS:
---
{context}
---

USER QUESTION: {request.question}

DETAILED ANSWER:"""
        
        answer = generate_response(prompt, max_tokens=800)
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "query"}
