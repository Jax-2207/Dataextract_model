"""
LLM integration for response generation
Supports both local Ollama and HuggingFace transformers
"""
import os
from typing import Optional

# Configuration
USE_OLLAMA = True  # Set to False to use HuggingFace transformers


def generate_response(prompt: str, max_tokens: int = 500) -> str:
    """
    Generate a response using LLM.
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated response text
    """
    if USE_OLLAMA:
        return generate_with_ollama(prompt, max_tokens)
    else:
        return generate_with_transformers(prompt, max_tokens)


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
                        "num_ctx": 4096,  # Reduced context for 4GB VRAM
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
                    return f"Error: Ollama returned status 500. Please restart Ollama: taskkill /IM ollama.exe /F && ollama serve"
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


def generate_with_transformers(prompt: str, max_tokens: int = 500) -> str:
    """
    Generate response using HuggingFace transformers.
    
    Note: This requires significant GPU memory for LLaMA models.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from app.config import LLM_MODEL_ID
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


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
