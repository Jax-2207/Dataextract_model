"""
OCR utilities using PaddleOCR (primary) and Tesseract (fallback)
"""
from typing import Optional, List, Dict, Any

# Lazy load PaddleOCR
_paddle_ocr = None


def get_paddle_ocr():
    """Get PaddleOCR instance (lazy loading)"""
    global _paddle_ocr
    
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except ImportError:
            print("Warning: PaddleOCR not installed. Use: pip install paddleocr")
            return None
    
    return _paddle_ocr


def ocr_image(img_path: str) -> str:
    """
    Extract text from image using PaddleOCR.
    
    Args:
        img_path: Path to image file
    
    Returns:
        Extracted text
    """
    ocr = get_paddle_ocr()
    
    if ocr is None:
        # Fallback to Tesseract if PaddleOCR not available
        return tesseract_ocr(img_path)
    
    try:
        result = ocr.ocr(img_path, cls=True)
        
        if result is None or result[0] is None:
            return ""
        
        # Extract text from results
        lines = []
        for line in result[0]:
            if line and len(line) > 1 and line[1]:
                text = line[1][0]  # Get text content
                lines.append(text)
        
        return " ".join(lines)
    except Exception as e:
        print(f"PaddleOCR error: {e}, falling back to Tesseract")
        return tesseract_ocr(img_path)


def ocr_image_detailed(img_path: str) -> List[Dict[str, Any]]:
    """
    Extract text with bounding boxes and confidence scores.
    
    Args:
        img_path: Path to image file
    
    Returns:
        List of dictionaries with text, box, and confidence
    """
    ocr = get_paddle_ocr()
    
    if ocr is None:
        return []
    
    try:
        result = ocr.ocr(img_path, cls=True)
        
        if result is None or result[0] is None:
            return []
        
        detailed_results = []
        for line in result[0]:
            if line and len(line) > 1:
                box = line[0]  # Bounding box coordinates
                text = line[1][0]  # Text content
                confidence = line[1][1]  # Confidence score
                
                detailed_results.append({
                    "text": text,
                    "box": box,
                    "confidence": confidence
                })
        
        return detailed_results
    except Exception as e:
        print(f"PaddleOCR error: {e}")
        return []


def tesseract_ocr(img_path: str) -> str:
    """
    Extract text using Tesseract OCR (fallback).
    
    Args:
        img_path: Path to image file
    
    Returns:
        Extracted text
    """
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image)
        
        return text.strip()
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""


def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """
    Extract text from scanned PDF using OCR.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text from all pages
    """
    try:
        from pdf2image import convert_from_path
        
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)
        
        all_text = []
        for i, page in enumerate(pages):
            # Save page as temp image
            temp_path = f"temp_page_{i}.png"
            page.save(temp_path, 'PNG')
            
            # OCR the page
            text = ocr_image(temp_path)
            all_text.append(text)
            
            # Cleanup temp file
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"Error extracting text from scanned PDF: {e}")
        return ""


def extract_text_from_image(image_path: str) -> str:
    """
    Main image text extraction function.
    Tries PaddleOCR first, falls back to Tesseract.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Extracted text
    """
    text = ocr_image(image_path)
    
    # If PaddleOCR returns empty, try Tesseract
    if not text.strip():
        text = tesseract_ocr(image_path)
    
    return text
