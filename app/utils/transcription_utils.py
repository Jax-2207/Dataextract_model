"""
Utilities for improving transcription quality
"""
import re


def clean_transcription(text: str) -> str:
    """
    Clean and format transcription text.
    
    Removes excessive whitespace, fixes punctuation, capitalizes sentences,
    and optionally removes filler words.
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned and formatted text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common transcription errors
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ?', '?')
    text = text.replace(' !', '!')
    text = text.replace('  ', ' ')
    
    # Capitalize sentences
    sentences = text.split('. ')
    sentences = [s.strip().capitalize() if s else s for s in sentences]
    text = '. '.join(sentences)
    
    # Remove common filler words (optional, can be disabled)
    fillers = [
        (' um ', ' '),
        (' uh ', ' '),
        (' like ', ' '),
        (' you know ', ' '),
        (' i mean ', ' ')
    ]
    for filler, replacement in fillers:
        text = text.replace(filler, replacement)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Remove any double spaces created
    text = text.strip()
    
    # Ensure proper ending
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text


def add_punctuation_hints(text: str) -> str:
    """
    Add punctuation based on common patterns.
    
    Args:
        text: Text without proper punctuation
    
    Returns:
        Text with improved punctuation
    """
    # Add periods at natural breaks (lowercase to uppercase transitions)
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    
    # Add question marks for question words
    question_words = r'\b(what|where|when|why|how|who|which|whose|whom)\b'
    text = re.sub(
        f'{question_words}([^.?!]+?)(?=[A-Z]|$)', 
        r'\1\2?', 
        text, 
        flags=re.IGNORECASE
    )
    
    return text


def format_timestamps(segments: list) -> str:
    """
    Format transcription segments with timestamps.
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text'
    
    Returns:
        Formatted text with timestamps
    """
    if not segments:
        return ""
    
    formatted_lines = []
    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        
        if text:
            # Format as [MM:SS] text
            start_min = int(start // 60)
            start_sec = int(start % 60)
            formatted_lines.append(f"[{start_min:02d}:{start_sec:02d}] {text}")
    
    return '\n'.join(formatted_lines)
