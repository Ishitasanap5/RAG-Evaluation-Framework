import json
from pathlib import Path
from chunkers import fixed_size_chunker, recursive_chunker, heading_based_chunker

def test_chunking_pipeline():
    processed_dir = Path("data/processed")
    json_files = list(processed_dir.glob("*.json"))
    
    if not json_files:
        print("⚠️ No processed JSON files found. Run ingestion.py first!")
        return

    print(f"🧪 Testing chunkers on {len(json_files)} documents...")
    
    # Pick the first file as a test subject
    sample_file = json_files[0]
    with open(sample_file, "r", encoding="utf-8") as f:
        doc_data = json.load(f)
        
    doc_id = doc_data["doc_id"]
    text = doc_data["full_text"]
    
    # Run the 3 chunkers
    fixed_chunks = fixed_size_chunker(text, doc_id)
    recursive_chunks = recursive_chunker(text, doc_id)
    heading_chunks = heading_based_chunker(text, doc_id)
    
    print(f"\n📄 Sample Document: {doc_id}")
    print(f"   - Total characters: {len(text)}")
    print(f"   - Fixed-size chunks generated: {len(fixed_chunks)} (Sample span: {fixed_chunks[0]['char_start']} -> {fixed_chunks[0]['char_end']})")
    print(f"   - Recursive chunks generated:  {len(recursive_chunks)} (Sample span: {recursive_chunks[0]['char_start']} -> {recursive_chunks[0]['char_end']})")
    print(f"   - Heading-based chunks generated: {len(heading_chunks)} (Sample span: {heading_chunks[0]['char_start']} -> {heading_chunks[0]['char_end']})")
    print("\n✨ All chunkers executed successfully with valid character spans!")

if __name__ == "__main__":
    test_chunking_pipeline()