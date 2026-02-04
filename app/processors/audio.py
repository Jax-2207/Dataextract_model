"""
Audio processing using OpenAI Whisper for transcription
"""
import os
from typing import Optional, Dict, Any

# Lazy load whisper model
_whisper_model = None


def get_whisper_model(model_size: str = "medium"):
    """Load Whisper model (lazy loading)"""
    global _whisper_model
    
    if _whisper_model is None:
        import whisper
        from app.config import WHISPER_MODEL
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    
    return _whisper_model


def audio_to_text(audio_path: str) -> str:
    """
    Transcribe audio file to text using Whisper.
    
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
    Transcribe audio with word-level timestamps.
    
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
    
    # Transcribe
    transcription = audio_to_text_with_timestamps(audio_path)
    result["text"] = transcription.get("text", "")
    result["segments"] = transcription.get("segments", [])
    result["language"] = transcription.get("language", "unknown")
    
    return result
