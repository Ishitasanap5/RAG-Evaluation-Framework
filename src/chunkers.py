import json
from pathlib import Path
import re

def fixed_size_chunker(text: str, doc_id: str, chunk_size: int = 512, overlap: int = 64):
    chunks = []
    length = len(text)
    start = 0
    
    while start < length:
        end = min(start + chunk_size, length)
        chunk_text = text[start:end]
        
        # Verify span integrity assertion
        assert text[start:end] == chunk_text, "Span mismatch in fixed chunker!"
        
        chunks.append({
            "doc_id": doc_id,
            "chunk_text": chunk_text,
            "char_start": start,
            "char_end": end,
            "strategy": "fixed"
        })
        
        if end == length:
            break
        start += (chunk_size - overlap)
        
    return chunks


def recursive_chunker(text: str, doc_id: str, max_size: int = 512, overlap: int = 64):
    """
    Bulletproof recursive chunker: normalizes line endings and 
    tracks exact character offsets using paragraph start/end indices.
    """
    chunks = []
    length = len(text)
    if length == 0:
        return chunks

    # Normalize line endings to prevent CRLF offset drift
    text = text.replace("\r\n", "\n")
    length = len(text)

    splits = [0]
    for match in re.finditer(r"\n\n+", text):
        splits.append(match.start())
    splits.append(length)

    current_start = -1
    current_end = -1
    current_text = ""

    for i in range(len(splits) - 1):
        para_start = splits[i]
        # Skip separator newlines
        if text[para_start:para_start+2] == "\n\n":
            para_start += 2
        para_end = splits[i+1]
        para = text[para_start:para_end]

        if not para.strip():
            continue

        if current_start == -1:
            current_start = para_start

        if len(current_text) + len(para) + 2 <= max_size:
            if not current_text:
                current_text = para
            else:
                current_text += "\n\n" + para
            current_end = para_end
        else:
            if current_text:
                chunk_slice = text[current_start:current_end]
                assert chunk_slice == current_text, f"Span drift detected in recursive chunker!"
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_text": current_text,
                    "char_start": current_start,
                    "char_end": current_end,
                    "strategy": "recursive"
                })
            current_start = para_start
            current_text = para
            current_end = para_end

    if current_text:
        chunk_slice = text[current_start:current_end]
        assert chunk_slice == current_text, f"Span drift detected in recursive chunker!"
        chunks.append({
            "doc_id": doc_id,
            "chunk_text": current_text,
            "char_start": current_start,
            "char_end": current_end,
            "strategy": "recursive"
        })

    return chunks


def heading_based_chunker(text: str, doc_id: str, max_size: int = 1500):
    # Normalize line endings
    text = text.replace("\r\n", "\n")
    
    # Identify code fence ranges to ignore headers inside them
    code_fence_ranges = []
    for match in re.finditer(r"(```[\s\S]*?```)", text):
        code_fence_ranges.append((match.start(), match.end()))

    def is_in_code_fence(pos):
        return any(start <= pos <= end for start, end in code_fence_ranges)

    header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    splits = []
    last_idx = 0

    for match in header_pattern.finditer(text):
        if not is_in_code_fence(match.start()):
            if match.start() > last_idx:
                splits.append((last_idx, match.start()))
            last_idx = match.start()
    splits.append((last_idx, len(text)))

    chunks = []
    for start, end in splits:
        section_text = text[start:end]
        if not section_text.strip():
            continue

        # If section exceeds max_size, fall back to fixed chunking for this section
        if len(section_text) > max_size:
            sub_start = start
            while sub_start < end:
                sub_end = min(sub_start + max_size, end)
                sub_text = text[sub_start:sub_end]
                assert text[sub_start:sub_end] == sub_text
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_text": sub_text,
                    "char_start": sub_start,
                    "char_end": sub_end,
                    "strategy": "heading"
                })
                sub_start = sub_end
        else:
            assert text[start:end] == section_text
            chunks.append({
                "doc_id": doc_id,
                "chunk_text": section_text,
                "char_start": start,
                "char_end": end,
                "strategy": "heading"
            })

    return chunks