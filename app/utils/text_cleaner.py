"""
Text cleaning utilities for preprocessing extracted content
"""
import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_headers_footers(text: str, patterns: Optional[list] = None) -> str:
    """
    Remove common header/footer patterns from document text.
    
    Args:
        text: Input text
        patterns: Optional list of regex patterns to remove
    
    Returns:
        Text with headers/footers removed
    """
    if patterns is None:
        # Common patterns
        patterns = [
            r'Page \d+ of \d+',
            r'Page \d+',
            r'^\d+$',  # Page numbers on their own line
            r'Confidential',
            r'All Rights Reserved',
        ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return clean_text(text)


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to their ASCII equivalents where possible.
    
    Args:
        text: Input text with unicode characters
    
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Common replacements
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u00ad': '',   # Soft hyphen
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def extract_sentences(text: str) -> list:
    """
    Split text into sentences.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence boundary detection
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: If True, keep basic punctuation
    
    Returns:
        Cleaned text
    """
    if keep_punctuation:
        # Keep letters, numbers, basic punctuation, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'"()-]', '', text)
    else:
        # Keep only letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return clean_text(text)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length, trying to break at word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum length of output
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncate_at = text.rfind(' ', 0, max_length - len(suffix))
    
    if truncate_at == -1:
        truncate_at = max_length - len(suffix)
    
    return text[:truncate_at] + suffix


def deduplicate_lines(text: str) -> str:
    """
    Remove duplicate consecutive lines.
    
    Args:
        text: Input text with potential duplicate lines
    
    Returns:
        Text with duplicates removed
    """
    lines = text.split('\n')
    result = []
    prev_line = None
    
    for line in lines:
        stripped = line.strip()
        if stripped != prev_line:
            result.append(line)
            prev_line = stripped
    
    return '\n'.join(result)
