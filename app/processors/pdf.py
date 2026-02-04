"""
PDF processing using PyMuPDF with OCR fallback for scanned documents
"""
import os
from typing import Dict, Any, Optional

import fitz  # PyMuPDF

from app.utils.ocr import ocr_image


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.
    Falls back to OCR for scanned pages.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text from all pages
    """
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page_num, page in enumerate(doc):
        # Try to extract text directly
        text = page.get_text()
        
        # If no text found, try OCR
        if not text.strip():
            text = ocr_pdf_page(page, page_num)
        
        all_text.append(text)
    
    doc.close()
    return "\n\n".join(all_text)


def ocr_pdf_page(page, page_num: int) -> str:
    """
    OCR a single PDF page.
    
    Args:
        page: PyMuPDF page object
        page_num: Page number (for temp file naming)
    
    Returns:
        OCR extracted text
    """
    # Render page to image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
    temp_img_path = f"temp_page_{page_num}.png"
    
    try:
        pix.save(temp_img_path)
        text = ocr_image(temp_img_path)
        return text
    finally:
        # Cleanup temp file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with PDF metadata
    """
    doc = fitz.open(pdf_path)
    
    metadata = {
        "page_count": doc.page_count,
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "producer": doc.metadata.get("producer", ""),
        "creation_date": doc.metadata.get("creationDate", ""),
        "modification_date": doc.metadata.get("modDate", ""),
        "is_encrypted": doc.is_encrypted,
        "is_pdf": doc.is_pdf,
    }
    
    # Get page dimensions from first page
    if doc.page_count > 0:
        first_page = doc[0]
        rect = first_page.rect
        metadata["page_width"] = rect.width
        metadata["page_height"] = rect.height
    
    doc.close()
    return metadata


def extract_images_from_pdf(pdf_path: str, output_dir: Optional[str] = None) -> list:
    """
    Extract all images from PDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
    
    Returns:
        List of paths to extracted images
    """
    import tempfile
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num, page in enumerate(doc):
        images = page.get_images()
        
        for img_num, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_path = os.path.join(
                output_dir, 
                f"page{page_num}_img{img_num}.{image_ext}"
            )
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_paths.append(image_path)
    
    doc.close()
    return image_paths


def extract_tables_from_pdf(pdf_path: str) -> list:
    """
    Extract tables from PDF (basic implementation).
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of tables (as list of rows)
    """
    doc = fitz.open(pdf_path)
    tables = []
    
    for page in doc:
        # Get all text blocks
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                # This is a simplified table detection
                # For better results, consider using Camelot or Tabula
                pass
    
    doc.close()
    return tables


def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Full PDF processing pipeline.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Dictionary with extracted text, metadata, and images
    """
    result = {
        "file_path": pdf_path,
        "file_name": os.path.basename(pdf_path),
        "type": "pdf"
    }
    
    # Extract metadata
    metadata = extract_pdf_metadata(pdf_path)
    result["metadata"] = metadata
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    result["text"] = text
    
    # Extract images (optional, can be resource intensive)
    # result["images"] = extract_images_from_pdf(pdf_path)
    
    return result
