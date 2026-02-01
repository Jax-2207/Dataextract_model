from fastapi import FastAPI
from app.api import upload

app = FastAPI(title="Multimodal RAG")

app.include_router(upload.router, prefix="/upload")

@app.get("/")
def home():
    return{"message" : "Multi-modal RAG backend is running"}