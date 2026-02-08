---
description: How to start the DataExtract RAG project
---

# DataExtract RAG - Startup Guide

## Prerequisites
- Python 3.10+ installed
- Ollama installed (https://ollama.com)
- MongoDB Atlas account (already configured)

## Step 1: Start Ollama (LLM)
```powershell
# In a new terminal window
ollama serve
```
Keep this terminal open.

## Step 2: Start the RAG Server
```powershell
# In another terminal, navigate to project
cd c:\Users\wania\OneDrive\Desktop\dataextract\Dataextract_model

# Activate virtual environment (if using one)
# .\venv\Scripts\activate

# Start the server
python -m uvicorn app.main:app --reload --port 8000
```

## Step 3: Open the UI
Open browser: **http://localhost:8000**

## Quick Start Commands (Copy-Paste)
```powershell
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Server
cd c:\Users\wania\OneDrive\Desktop\dataextract\Dataextract_model
python -m uvicorn app.main:app --reload --port 8000
```

## Verify Everything Works
- Dashboard: http://localhost:8000 → Dashboard page shows all green ✅
- Upload: Upload a PDF/document
- Query: Ask questions about your documents

## Troubleshooting
| Issue | Solution |
|-------|----------|
| "Ollama not found" | Restart terminal after installing Ollama |
| "CUDA error" | Run `taskkill /IM ollama.exe /F` then `ollama serve` |
| "Port 8000 in use" | Use `--port 8001` instead |
