import pytest
from src.chunkers import fixed_size_chunker, recursive_chunker, heading_based_chunker

def test_fixed_chunker():
    text = "Hello world! " * 100
    chunks = fixed_size_chunker(text, "test.md", chunk_size=100, overlap=20)
    assert len(chunks) > 0
    for c in chunks:
        assert text[c["char_start"]:c["char_end"]] == c["chunk_text"]

def test_recursive_chunker():
    text = "Paragraph A.\n\nParagraph B is much longer and contains several sentences to test paragraph breaking.\n\nParagraph C."
    chunks = recursive_chunker(text, "test.md", max_size=50, overlap=10)
    assert len(chunks) > 0
    for c in chunks:
        assert text[c["char_start"]:c["char_end"]] == c["chunk_text"]

def test_heading_chunker():
    text = "# Main Title\n\nIntro paragraph.\n\n```python\n# Code comment header simulation\ndef foo(): pass\n```\n\n## Subtitle\n\nBody content under subtitle."
    chunks = heading_based_chunker(text, "test.md", max_size=1000)
    assert len(chunks) > 0
    for c in chunks:
        assert text[c["char_start"]:c["char_end"]] == c["chunk_text"]