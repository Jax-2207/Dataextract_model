"""
Utility functions for ensuring diversity in retrieved chunks
"""
from typing import List, Dict, Any


def ensure_file_diversity(
    chunks: List[Dict[str, Any]], 
    max_chunks: int = 10,
    max_per_file: int = 5
) -> List[Dict[str, Any]]:
    """
    Ensure diversity across different source files in retrieved chunks.
    
    Uses round-robin selection to distribute chunks across multiple files,
    preventing dominance by a single document.
    
    Args:
        chunks: List of retrieved chunks (ordered by relevance)
        max_chunks: Maximum number of chunks to return
        max_per_file: Maximum chunks from a single file
    
    Returns:
        Diversified list of chunks maintaining relevance order within files
    """
    if not chunks:
        return []
    
    # Group chunks by file, preserving order within each file
    file_groups = {}
    for chunk in chunks:
        file_path = chunk.get('file_path', chunk.get('file', 'unknown'))
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(chunk)
    
    # If only one file, just return top chunks
    if len(file_groups) == 1:
        return chunks[:max_chunks]
    
    # Round-robin selection from different files
    result = []
    file_paths = list(file_groups.keys())
    per_file_count = {fp: 0 for fp in file_paths}
    
    # Keep track of position in each file's chunk list
    file_positions = {fp: 0 for fp in file_paths}
    
    while len(result) < max_chunks:
        added_in_round = False
        
        for file_path in file_paths:
            if len(result) >= max_chunks:
                break
                
            # Check if we can add from this file
            if (per_file_count[file_path] < max_per_file and 
                file_positions[file_path] < len(file_groups[file_path])):
                
                chunk = file_groups[file_path][file_positions[file_path]]
                result.append(chunk)
                per_file_count[file_path] += 1
                file_positions[file_path] += 1
                added_in_round = True
        
        # If no chunks were added in this round, we're done
        if not added_in_round:
            break
    
    return result
