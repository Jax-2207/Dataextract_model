import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def extract_text_from_scanned_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""

    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"

    return text


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)
