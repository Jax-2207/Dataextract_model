"""
Test script for deleted chunk filtering
"""
import sys
sys.path.insert(0, '.')

from app.storage.vector_store import get_chunks_by_indices, get_chunk_mapping
import numpy as np

print("Testing deleted chunk filtering...")
print("=" * 50)

mapping = get_chunk_mapping()
print(f"Total chunks in mapping: {len(mapping)}")

if mapping:
    # Get first 5 indices
    indices = np.array(list(mapping.keys())[:5])
    chunks = get_chunks_by_indices(indices)
    
    deleted_count = sum(1 for c in chunks if c.get('deleted', False))
    
    print(f"Retrieved {len(chunks)} chunks")
    print(f"Deleted chunks in result: {deleted_count}")
    
    assert deleted_count == 0, f"Found {deleted_count} deleted chunks!"
    print("\n✓ Deleted chunks are properly filtered")
else:
    print("No chunks in mapping yet (empty database)")
    print("✓ Test skipped - database is empty")

print("=" * 50)
