import json
from pathlib import Path
from chunkers import heading_based_chunker

def inspect():
    path = Path("data/processed/alternatives.md.json")
    if not path.exists():
        print("Processed file not found.")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    chunks = heading_based_chunker(data["full_text"], data["doc_id"])
    print(f"Total chunks in alternatives.md: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Span: {c['char_start']} -> {c['char_end']}")
        print(f"Snippet: {c['chunk_text'][:150]}...")

if __name__ == "__main__":
    inspect()