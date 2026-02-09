"""
LLM integration for response generation
Supports Groq API (cloud) and local Ollama
"""
import os
from typing import Optional

from app.config import USE_CLOUD_LLM, GROQ_API_KEY, GROQ_LLM_MODEL


def generate_response(prompt: str, max_tokens: int = 1000) -> str:
    """
    Generate a response using LLM.
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated response text
    """
    if USE_CLOUD_LLM:
        return generate_with_groq(prompt, max_tokens)
    else:
        return generate_with_ollama(prompt, max_tokens)


def generate_with_groq(prompt: str, max_tokens: int = 1000) -> str:
    """
    Generate response using Groq API (FREE, ultra-fast).
    
    Uses llama-3.3-70b-versatile - much more capable than local 3B models.
    Free tier: 14,400 requests/day
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Use chat completion for better responses
        response = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a RAG (Retrieval-Augmented Generation) assistant for an educational platform.

CRITICAL RULES:
1. ONLY answer based on the context/documents provided in the user's message
2. DO NOT use your general knowledge or training data
3. If the context doesn't contain the answer, say "I couldn't find this in your uploaded documents"
4. Always reference the source material when answering
5. Format responses clearly with markdown when appropriate"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            print("Groq rate limit hit, falling back to local Ollama...")
            return generate_with_ollama(prompt, max_tokens)
        return f"Error with Groq API: {error_msg}"


def generate_with_ollama(prompt: str, max_tokens: int = 500) -> str:
    """
    Generate response using Ollama (local LLaMA)
    
    Includes retry logic for GPU memory errors.
    Requires Ollama to be running locally.
    """
    import requests
    import time
    from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            elif response.status_code == 500:
                error_text = response.text
                if "CUDA" in error_text or "memory" in error_text.lower():
                    print(f"GPU memory error on attempt {attempt + 1}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    return f"Error: Ollama returned status 500. Please restart Ollama."
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running (ollama serve)"
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(retry_delay)
                continue
            return "Error: Ollama request timed out. The model might be overloaded."
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return "Error: Failed after multiple retries. Please restart Ollama and try again."


def generate_with_groq_chat(messages: list, max_tokens: int = 1000) -> str:
    """
    Chat-style generation with Groq (supports conversation history)
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error with Groq chat: {str(e)}"


def generate_with_ollama_chat(messages: list, max_tokens: int = 500) -> str:
    """
    Chat-style generation with Ollama (supports conversation history)
    """
    try:
        import requests
        from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        else:
            return f"Error: Ollama returned status {response.status_code}"
            
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_with_confidence(prompt: str, context: str, question: str, max_tokens: int = 1000) -> dict:
    """
    Generate a response with confidence score.
    
    Returns:
        dict with 'answer', 'confidence_score', and 'reasoning'
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        confidence_prompt = f"""You are a RAG assistant. Answer the question based ONLY on the provided context.

CONTEXT FROM UPLOADED FILES:
---
{context}
---

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY the context above
2. Rate your confidence (0-100%) based on how well the context answers the question
3. If the context doesn't contain relevant information, confidence should be LOW (below 30%)

Respond in this EXACT format:
ANSWER: [Your answer here]
CONFIDENCE: [0-100]
REASONING: [Why you gave this confidence score]"""

        response = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers with confidence scores."},
                {"role": "user", "content": confidence_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        
        result_text = response.choices[0].message.content
        return parse_confidence_response(result_text)
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence_score": 0,
            "reasoning": "Error occurred during generation"
        }


def generate_internet_answer(question: str, max_tokens: int = 1000) -> dict:
    """
    Generate an answer using the LLM's general knowledge (internet mode).
    
    Returns:
        dict with 'answer', 'confidence_score', and 'reasoning'
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        internet_prompt = f"""You are a knowledgeable educational assistant. Answer the following question using your training knowledge.

QUESTION: {question}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer
2. Rate your confidence (0-100%) based on how certain you are of the accuracy
3. Only give high confidence (90%+) if you're very sure the information is accurate

Respond in this EXACT format:
ANSWER: [Your detailed answer here]
CONFIDENCE: [0-100]
REASONING: [Why you gave this confidence score]"""

        response = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant with broad knowledge."},
                {"role": "user", "content": internet_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        
        result_text = response.choices[0].message.content
        return parse_confidence_response(result_text)
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "confidence_score": 0,
            "reasoning": "Error occurred during generation"
        }


def parse_confidence_response(text: str) -> dict:
    """
    Parse the structured response with answer and confidence.
    """
    import re
    
    result = {
        "answer": "",
        "confidence_score": 50,
        "reasoning": ""
    }
    
    # Extract ANSWER
    answer_match = re.search(r'ANSWER:\s*(.+?)(?=CONFIDENCE:|$)', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    else:
        # If no format found, use the whole text as answer
        result["answer"] = text.strip()
    
    # Extract CONFIDENCE
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)
    if confidence_match:
        result["confidence_score"] = min(100, max(0, int(confidence_match.group(1))))
    
    # Extract REASONING
    reasoning_match = re.search(r'REASONING:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    return result

