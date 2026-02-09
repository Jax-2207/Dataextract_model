"""
RAG Query API endpoint
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.core.embeddings import get_query_embeddings
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
        # Step 1: Embed the question using query-optimized embeddings
        question_embedding = get_query_embeddings(request.question)
        
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
        
        # Step 5: Generate response using LLM with strict context-only prompt
        prompt = f"""You are a helpful teacher assistant on an educational platform.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY use information from the "CONTEXT FROM UPLOADED FILES" section below
2. DO NOT use any external knowledge or information from the internet
3. If the answer is not found in the context, say: "I couldn't find information about this in your uploaded documents. Please upload relevant materials."
4. Always cite which part of the uploaded content you're referencing

FORMATTING RULES:
1. **For code**: Use markdown code blocks with language specification
2. **For math**: Show step-by-step solutions
3. **Use bullet points** and numbered lists for clarity
4. **Bold** important terms
5. Be friendly and conversational

CONTEXT FROM UPLOADED FILES:
---
{context}
---

STUDENT'S QUESTION: {request.question}

YOUR RESPONSE (ONLY use the context above, do NOT use external knowledge):"""
        
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
