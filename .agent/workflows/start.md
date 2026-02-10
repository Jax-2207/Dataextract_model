---
description: How to start the DataExtract RAG project
---

# DataExtract RAG - Startup Guide

## Prerequisites
- Python 3.10+ installed
- Groq API key (free: https://console.groq.com)
- Cohere API key (free: https://dashboard.cohere.com)
- MongoDB Atlas account (already configured)

## Step 1: Set up environment variables
Copy `.env.example` to `.env` and fill in your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=YourApp
MONGODB_DB_NAME=multimodal_rag
USE_CLOUD_LLM=true
USE_CLOUD_WHISPER=true
USE_CLOUD_EMBEDDINGS=true
```

## Step 2: Install dependencies
```powershell
cd c:\Users\wania\OneDrive\Desktop\dataextract\Dataextract_model
pip install -r requirements.txt
```

## Step 3: Start the RAG Server
```powershell
cd c:\Users\wania\OneDrive\Desktop\dataextract\Dataextract_model
python -m uvicorn app.main:app --reload --port 8000
```

## Step 4: Open the UI
Open browser: **http://localhost:8000**

## Quick Start Commands (Copy-Paste)
```powershell
# Navigate to project
cd c:\Users\wania\OneDrive\Desktop\dataextract\Dataextract_model

# Install deps (first time only)
pip install -r requirements.txt

# Start Server
python -m uvicorn app.main:app --reload --port 8000
```

## Tech Stack
| Component | Provider | Details |
|-----------|----------|---------|
| LLM | **Groq** | llama-3.3-70b-versatile (14,400 req/day free) |
| Whisper | **Groq** | whisper-large-v3 |
| Embeddings | **Cohere** | embed-english-v3.0 (1000 req/min free) |
| Vector Store | **FAISS** | Local, 1024-dim vectors |
| Database | **MongoDB Atlas** | Cloud metadata storage |
| File Storage | **Cloudinary** (optional) | 25GB free |
| OCR | **PaddleOCR** / Tesseract | Local |

## Verify Everything Works
- Dashboard: http://localhost:8000 → Dashboard page shows all green ✅
- Upload: Upload a PDF/document
- Query: Ask questions about your documents
- API Docs: http://localhost:8000/docs

## Troubleshooting
| Issue | Solution |
|-------|----------|
| "GROQ_API_KEY not set" | Check your `.env` file has a valid key |
| "Cohere error" | Verify `COHERE_API_KEY` in `.env` |
| "MongoDB connection failed" | Check `MONGODB_URI` in `.env` |
| "Port 8000 in use" | Use `--port 8001` instead |
