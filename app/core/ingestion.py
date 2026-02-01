import os
from app.core.router import detect_file_type
from app.processors.pdf import extract_text_from_pdf

UPLOAD_DIR = "data/uploads"

def ingest_file(file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    file_type = detect_file_type(file.filename)

    extracted_text = ""

    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(file_path)

    return file_path, file_type, extracted_text
