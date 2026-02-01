from fastapi import APIRouter, UploadFile, File
import os
import shutil

from app.processors.pdf import extract_text_from_pdf
from app.processors.ocr import (
    extract_text_from_scanned_pdf,
    extract_text_from_image
)

router = APIRouter()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = file.filename.split(".")[-1].lower()

    # -------- PDF --------
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)

        if text.strip() == "":
            text = extract_text_from_scanned_pdf(file_path)

        return {
            "file_type": "pdf",
            "extracted_text_preview": text[:500]
        }

    # -------- IMAGE --------
    elif ext in ["jpg", "jpeg", "png"]:
        text = extract_text_from_image(file_path)

        return {
            "file_type": "image",
            "extracted_text_preview": text[:500]
        }

    else:
        return {"message": "Unsupported file type"}
