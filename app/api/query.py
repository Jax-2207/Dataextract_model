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
        
        # Step 5: Generate response using LLM with teacher-style prompt
        prompt = f"""You are a friendly, patient, and expert teacher on an educational platform. Think like ChatGPT or Gemini - be conversational, clear, and helpful.

FORMATTING RULES (IMPORTANT):
1. **For coding questions**: Always use markdown code blocks with language specification:
   ```python
   print("Hello World")
   ```
2. **For math/equations**: Use clear notation like: x² + 2x + 1 = 0, or step-by-step:
   Step 1: ...
   Step 2: ...
3. **Use bullet points** and numbered lists for clarity
4. **Bold** important terms and concepts
5. **Break down complex topics** into simple, digestible parts
6. Be warm, encouraging, and conversational - like a friend who's great at teaching

YOUR TEACHING APPROACH:
- Start with a brief, friendly acknowledgment of the question
- Explain concepts step-by-step with examples
- Use analogies when helpful
- End with encouragement or a tip for further learning
- If the uploaded content covers the topic, use it. Otherwise, teach from your knowledge and mention you're adding context

LEARNING MATERIALS FROM STUDENT'S UPLOADS:
---
{context}
---

STUDENT'S QUESTION: {request.question}

YOUR RESPONSE (be like ChatGPT - friendly, well-formatted, easy to understand):"""
        
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
