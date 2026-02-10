# Multimodal RAG System

A powerful Retrieval-Augmented Generation (RAG) system that supports multiple file types including PDFs, images, audio, and video.

## 🚀 Tech Stack

| Layer | Technology |
|-------|------------|
| Backend API | FastAPI |
| Language | Python |
| OCR | PaddleOCR (+ Tesseract fallback) |
| Documents | PyMuPDF, Apache Tika |
| Audio/Video | Whisper, FFmpeg |
| Vision | OpenCV, BLIP, OpenCLIP |
| Embeddings | BGE / MiniLM |
| Vector DB | FAISS |
| LLM | LLaMA 3.1 (local via Ollama) |
| Metadata DB | PostgreSQL |

## 📁 Project Structure

```
multimodal-rag/
│
├── app/
│   ├── main.py                  # FastAPI entry
│   ├── config.py                # Configuration settings
│   │
│   ├── api/
│   │   ├── upload.py            # File upload endpoints
│   │   └── query.py             # RAG query endpoints
│   │
│   ├── core/
│   │   ├── router.py            # File type detection
│   │   ├── ingestion.py         # Pipeline orchestration
│   │   ├── chunking.py          # Text chunking
│   │   ├── embeddings.py        # Embedding generation
│   │   └── llm.py               # LLM integration
│   │
│   ├── processors/
│   │   ├── pdf.py               # PDF processing
│   │   ├── audio.py             # Audio transcription
│   │   ├── video.py             # Video processing
│   │   └── image.py             # Image processing
│   │
│   ├── storage/
│   │   ├── vector_store.py      # FAISS vector store
│   │   └── metadata_db.py       # PostgreSQL metadata
│   │
│   └── utils/
│       ├── ocr.py               # OCR utilities
│       └── text_cleaner.py      # Text preprocessing
│
├── data/
│   ├── uploads/                 # Uploaded files
│   ├── processed/               # Processed data
│   └── faiss/                   # FAISS index storage
│
├── requirements.txt
└── README.md
```

## 🛠️ Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Additional Requirements

**FFmpeg** (for audio/video processing):
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

**Tesseract OCR** (fallback OCR):
- Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt install tesseract-ocr`
- Mac: `brew install tesseract`

**Ollama** (for local LLM):
- Download from https://ollama.ai
- Pull LLaMA model: `ollama pull llama3.1`

### 4. Setup PostgreSQL (Optional)

```bash
# Create database
createdb multimodal_rag

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=multimodal_rag
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
```

## 🚀 Running the Application

### Start Ollama (for LLM)
```bash
ollama serve
```

### Start the API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the API
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## 📝 API Endpoints

### Upload Files

```bash
# Upload single file
curl -X POST "http://localhost:8000/upload/" \
  -H "accept: application/json" \
  -F "file=@document.pdf"

# Simple extraction (no vector storage)
curl -X POST "http://localhost:8000/upload/simple" \
  -H "accept: application/json" \
  -F "file=@image.png"
```

### Query

```bash
# Ask a question
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

## 📂 Supported File Types

| Type | Extensions |
|------|------------|
| PDF | .pdf |
| Images | .jpg, .jpeg, .png |
| Audio | .mp3, .wav |
| Video | .mp4 |

## 🔧 Configuration

Edit `app/config.py` to customize:

- **Embedding model**: Change `EMBEDDING_MODEL`
- **Chunk size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **LLM model**: Update `OLLAMA_MODEL`
- **OCR settings**: Toggle `USE_PADDLEOCR`

## 📊 Features

- ✅ Multi-format document processing
- ✅ OCR for scanned documents (PaddleOCR + Tesseract)
- ✅ Audio/Video transcription (Whisper)
- ✅ Image captioning (BLIP)
- ✅ Semantic chunking with overlap
- ✅ BGE embeddings for semantic search
- ✅ FAISS vector storage
- ✅ Local LLM integration (Ollama)
- ✅ PostgreSQL metadata tracking
- ✅ RESTful API with FastAPI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License
