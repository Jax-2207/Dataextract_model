"""
Video processing using FFmpeg, Whisper, and OpenCV
Extracts audio transcription and key frames
"""
import os
import subprocess
import tempfile
from typing import List, Dict, Any, Optional

from app.processors.audio import audio_to_text, audio_to_text_with_timestamps


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from video using FFmpeg.
    
    Args:
        video_path: Path to video file
        output_path: Optional output path for audio file
    
    Returns:
        Path to extracted audio file
    """
    if output_path is None:
        # Create temp file for audio
        output_path = tempfile.mktemp(suffix='.wav')
    
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # WAV format
        '-ar', '16000',  # 16kHz sample rate (good for Whisper)
        '-ac', '1',  # Mono
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    
    return output_path


def extract_key_frames(
    video_path: str, 
    output_dir: Optional[str] = None,
    interval_seconds: float = 5.0
) -> List[str]:
    """
    Extract key frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        interval_seconds: Extract one frame every N seconds
    
    Returns:
        List of paths to extracted frame images
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'fps=1/{interval_seconds}',  # Extract frame every N seconds
        '-q:v', '2',  # High quality JPEG
        output_pattern
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
    
    # Get list of extracted frames
    frames = sorted([
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.startswith('frame_') and f.endswith('.jpg')
    ])
    
    return frames


def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extract metadata from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video metadata
    """
    try:
        import json
        
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            return {
                "duration": float(format_info.get('duration', 0)),
                "size": int(format_info.get('size', 0)),
                "bitrate": int(format_info.get('bit_rate', 0)),
                "format": format_info.get('format_name', 'unknown'),
                "width": video_stream.get('width', 0) if video_stream else 0,
                "height": video_stream.get('height', 0) if video_stream else 0,
                "fps": eval(video_stream.get('r_frame_rate', '0/1')) if video_stream else 0
            }
        
        return {}
        
    except Exception as e:
        return {"error": str(e)}


def video_to_text(video_path: str) -> str:
    """
    Transcribe video audio to text.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Transcribed text from video audio
    """
    # Extract audio
    audio_path = extract_audio_from_video(video_path)
    
    try:
        # Transcribe
        text = audio_to_text(audio_path)
        return text
    finally:
        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)


def process_video(
    video_path: str,
    extract_frames: bool = True,
    frame_interval: float = 10.0
) -> Dict[str, Any]:
    """
    Full video processing pipeline.
    
    Args:
        video_path: Path to video file
        extract_frames: Whether to extract key frames
        frame_interval: Interval in seconds between frame extractions
    
    Returns:
        Dictionary with transcription, frames, and metadata
    """
    result = {
        "file_path": video_path,
        "file_name": os.path.basename(video_path),
        "type": "video"
    }
    
    # Get metadata
    metadata = extract_video_metadata(video_path)
    result["metadata"] = metadata
    
    # Extract and transcribe audio
    try:
        audio_path = extract_audio_from_video(video_path)
        transcription = audio_to_text_with_timestamps(audio_path)
        result["text"] = transcription.get("text", "")
        result["segments"] = transcription.get("segments", [])
        result["language"] = transcription.get("language", "unknown")
        
        # Cleanup temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception as e:
        result["text"] = ""
        result["error"] = str(e)
    
    # Extract key frames if requested
    if extract_frames:
        try:
            frames = extract_key_frames(video_path, interval_seconds=frame_interval)
            result["frames"] = frames
            result["frame_count"] = len(frames)
        except Exception as e:
            result["frames"] = []
            result["frame_error"] = str(e)
    
    return result
