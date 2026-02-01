document_store = []

from fastapi import FastAPI, UploadFile, File
import os, shutil

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = ""

    #  PDF FILE
    if file.content_type == "application/pdf":
        pdf = fitz.open(file_path)
        for page in pdf:
            text += page.get_text()

    #  IMAGE FILE
    elif file.content_type.startswith("image/"):
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

    else:
        return {"error": "Unsupported file type"}

    # ADD THIS PART HERE
    document_store.append({
        "filename": file.filename,
        "content": text
    })

    return {
        "message": "Text extracted and stored",
        "filename": file.filename
    }

from fastapi import Query

@app.get("/search/")
def search_text(q: str = Query(...)):
    results = []

    for doc in document_store:
        if q.lower() in doc["content"].lower():
            results.append({
                "filename": doc["filename"],
                "text": doc["content"][:300]
            })

    return results
