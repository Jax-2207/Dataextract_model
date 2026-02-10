"""
Test script for file diversity functionality
"""
from app.utils.diversity import ensure_file_diversity

# Test 1: Basic diversity
print("Test 1: Basic file diversity")
chunks = [
    {'text': 'chunk1', 'file_path': 'file1.pdf'},
    {'text': 'chunk2', 'file_path': 'file1.pdf'},
    {'text': 'chunk3', 'file_path': 'file2.pdf'},
    {'text': 'chunk4', 'file_path': 'file1.pdf'},
    {'text': 'chunk5', 'file_path': 'file2.pdf'},
]

result = ensure_file_diversity(chunks, max_chunks=4, max_per_file=2)
files = [c['file_path'] for c in result]

print(f"  Result files: {files}")
print(f"  File1 count: {files.count('file1.pdf')}")
print(f"  File2 count: {files.count('file2.pdf')}")

assert 'file1.pdf' in files and 'file2.pdf' in files, "Missing files!"
assert files.count('file1.pdf') <= 2, "Too many from file1!"
assert files.count('file2.pdf') <= 2, "Too many from file2!"
print("  ✓ Test 1 passed\n")

# Test 2: Single file
print("Test 2: Single file (no diversity needed)")
chunks = [
    {'text': 'chunk1', 'file_path': 'file1.pdf'},
    {'text': 'chunk2', 'file_path': 'file1.pdf'},
]
result = ensure_file_diversity(chunks, max_chunks=2)
print(f"  Result count: {len(result)}")
assert len(result) == 2, "Should return all chunks for single file"
print("  ✓ Test 2 passed\n")

# Test 3: Empty chunks
print("Test 3: Empty chunks")
result = ensure_file_diversity([], max_chunks=5)
assert len(result) == 0, "Should return empty list"
print("  ✓ Test 3 passed\n")

print("=" * 50)
print("✓ All file diversity tests passed!")
print("=" * 50)
