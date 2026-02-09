"""
Audio processing with Groq Whisper API (cloud) and local Whisper fallback
"""
import os
from typing import Optional, Dict, Any

from app.config import USE_CLOUD_WHISPER, GROQ_API_KEY, GROQ_WHISPER_MODEL

# Lazy load whisper model (for local fallback)
_whisper_model = None


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio file to text.
    Uses Groq Whisper API (cloud) or local Whisper based on config.
    
    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
    
    Returns:
        Dictionary with text, segments, and language
    """
    if USE_CLOUD_WHISPER:
        return transcribe_with_groq(audio_path)
    else:
        return audio_to_text_with_timestamps(audio_path)


def transcribe_with_groq(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio using Groq Whisper API (FREE, fast, accurate).
    
    Uses whisper-large-v3 - much more accurate than local tiny model.
    Free tier included in Groq's 14,400 requests/day.
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        # Read audio file
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=GROQ_WHISPER_MODEL,
                file=audio_file,
                response_format="verbose_json",  # Get timestamps
                language="en"  # Can be made dynamic
            )
        
        # Parse response
        result = {
            "text": transcription.text,
            "language": getattr(transcription, 'language', 'en'),
            "duration": getattr(transcription, 'duration', 0),
            "segments": []
        }
        
        # Extract segments if available
        if hasattr(transcription, 'segments') and transcription.segments:
            result["segments"] = [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "")
                }
                for seg in transcription.segments
            ]
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"Groq Whisper error: {error_msg}")
        
        # Check file size - Groq has 25MB limit
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 25:
            print(f"Audio file is {file_size_mb:.1f}MB (>25MB limit). Falling back to local Whisper...")
            return audio_to_text_with_timestamps(audio_path)
        
        # Rate limit or other error - fallback to local
        if "rate_limit" in error_msg.lower():
            print("Groq rate limit hit, falling back to local Whisper...")
            return audio_to_text_with_timestamps(audio_path)
        
        return {"error": error_msg, "text": ""}


def get_whisper_model(model_size: str = "medium"):
    """Load local Whisper model (lazy loading) - forces CPU for low VRAM systems"""
    global _whisper_model
    
    if _whisper_model is None:
        import whisper
        import torch
        from app.config import WHISPER_MODEL
        
        # Force CPU to avoid GPU memory issues on low VRAM systems
        device = "cpu"
        print(f"Loading local Whisper model '{WHISPER_MODEL}' on {device}...")
        _whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        print("Whisper model loaded successfully!")
    
    return _whisper_model


def audio_to_text(audio_path: str) -> str:
    """
    Transcribe audio file to text using local Whisper.
    
    Args:
        audio_path: Path to audio file (mp3, wav, etc.)
    
    Returns:
        Transcribed text
    """
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"


def audio_to_text_with_timestamps(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio with word-level timestamps using local Whisper.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with text and segments with timestamps
    """
    try:
        model = get_whisper_model()
        result = model.transcribe(audio_path, word_timestamps=True)
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0)
        }
    except Exception as e:
        return {"error": str(e), "text": ""}


def extract_audio_metadata(audio_path: str) -> Dict[str, Any]:
    """
    Extract metadata from audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with audio metadata
    """
    try:
        import subprocess
        import json
        
        # Use ffprobe to get metadata
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            
            return {
                "duration": float(format_info.get('duration', 0)),
                "size": int(format_info.get('size', 0)),
                "bitrate": int(format_info.get('bit_rate', 0)),
                "format": format_info.get('format_name', 'unknown')
            }
        
        return {}
        
    except Exception as e:
        return {"error": str(e)}


def process_audio(audio_path: str) -> Dict[str, Any]:
    """
    Full audio processing pipeline.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with transcription and metadata
    """
    result = {
        "file_path": audio_path,
        "file_name": os.path.basename(audio_path),
        "type": "audio"
    }
    
    # Get metadata
    metadata = extract_audio_metadata(audio_path)
    result["metadata"] = metadata
    
    # Transcribe using cloud or local
    transcription = transcribe_audio(audio_path)
    result["text"] = transcription.get("text", "")
    result["segments"] = transcription.get("segments", [])
    result["language"] = transcription.get("language", "unknown")
    
    if transcription.get("error"):
        result["error"] = transcription["error"]
    
    return result
