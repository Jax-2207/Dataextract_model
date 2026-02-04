"""
Image processing using OpenCV, BLIP, and OCR
Extracts text and generates captions
"""
import os
from typing import Dict, Any, Optional, List

# Lazy load models
_blip_processor = None
_blip_model = None
_clip_model = None
_clip_processor = None


def get_blip_model():
    """Load BLIP model for image captioning (lazy loading)"""
    global _blip_processor, _blip_model
    
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    return _blip_processor, _blip_model


def get_clip_model():
    """Load CLIP model for image embeddings (lazy loading)"""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    return _clip_processor, _clip_model


def generate_caption(image_path: str) -> str:
    """
    Generate a caption for an image using BLIP.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Generated caption text
    """
    try:
        from PIL import Image
        
        processor, model = get_blip_model()
        
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        output = model.generate(**inputs, max_length=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"


def get_image_embedding(image_path: str):
    """
    Get CLIP embedding for an image.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image embedding vector
    """
    try:
        from PIL import Image
        import torch
        
        processor, model = get_clip_model()
        
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        return image_features.numpy().flatten()
    except Exception as e:
        print(f"Error getting image embedding: {e}")
        return None


def extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extract metadata from image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with image metadata
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        image = Image.open(image_path)
        
        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode
        }
        
        # Extract EXIF data if available
        exif_data = image._getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    continue  # Skip binary data
                exif[tag] = str(value)
            metadata["exif"] = exif
        
        return metadata
    except Exception as e:
        return {"error": str(e)}


def preprocess_image(image_path: str) -> str:
    """
    Preprocess image for better OCR results.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Path to preprocessed image
    """
    try:
        import cv2
        import numpy as np
        
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Save preprocessed image
        preprocessed_path = image_path.replace('.', '_preprocessed.')
        cv2.imwrite(preprocessed_path, denoised)
        
        return preprocessed_path
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image_path


def extract_text_from_image(image_path: str, use_paddleocr: bool = True) -> str:
    """
    Extract text from image using OCR.
    
    Args:
        image_path: Path to image file
        use_paddleocr: If True, use PaddleOCR; else use Tesseract
    
    Returns:
        Extracted text
    """
    from app.utils.ocr import ocr_image, tesseract_ocr
    
    if use_paddleocr:
        text = ocr_image(image_path)
        # Fallback to Tesseract if PaddleOCR fails or returns empty
        if not text.strip():
            text = tesseract_ocr(image_path)
    else:
        text = tesseract_ocr(image_path)
    
    return text


def process_image(image_path: str, extract_text: bool = True, generate_caption_flag: bool = True) -> Dict[str, Any]:
    """
    Full image processing pipeline.
    
    Args:
        image_path: Path to image file
        extract_text: Whether to extract text using OCR
        generate_caption_flag: Whether to generate image caption
    
    Returns:
        Dictionary with extracted text, caption, and metadata
    """
    result = {
        "file_path": image_path,
        "file_name": os.path.basename(image_path),
        "type": "image"
    }
    
    # Get metadata
    metadata = extract_image_metadata(image_path)
    result["metadata"] = metadata
    
    # Extract text using OCR
    if extract_text:
        try:
            text = extract_text_from_image(image_path)
            result["text"] = text
        except Exception as e:
            result["text"] = ""
            result["ocr_error"] = str(e)
    
    # Generate caption
    if generate_caption_flag:
        try:
            caption = generate_caption(image_path)
            result["caption"] = caption
        except Exception as e:
            result["caption"] = ""
            result["caption_error"] = str(e)
    
    return result
