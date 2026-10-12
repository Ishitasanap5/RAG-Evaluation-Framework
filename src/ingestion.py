import json
from pathlib import Path

# Files that make poor eval targets / flood the index with near-duplicate text.
# _llm-test.md is a translation test page; release-notes.md is ~500K chars of changelog.
# Remove entries here if you want them in the corpus.
DEFAULT_EXCLUDE = {"_llm-test.md", "release-notes.md"}


def ingest_corpus(raw_dir: str = "data/raw", processed_dir: str = "data/processed", exclude=DEFAULT_EXCLUDE):
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    md_files = sorted(raw_path.glob("**/*.md"))
    if not md_files:
        print(f"No markdown files found in {raw_dir}")
        return

    # Remove stale output (including nested folders created by the old Windows bug)
    for old in processed_path.rglob("*.json"):
        old.unlink()

    print("Starting corpus ingestion...")
    count, skipped = 0, 0
    for file_path in md_files:
        # as_posix() => forward slashes on every OS, e.g. "tutorial/first-steps.md"
        doc_id = file_path.relative_to(raw_path).as_posix()
        if doc_id in exclude:
            skipped += 1
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1")

        cleaned = content.replace("\r\n", "\n")
        doc_data = {"doc_id": doc_id, "full_text": cleaned, "char_length": len(cleaned)}

        # Flat filename, so the downstream glob("*.json") sees every document
        out_file = processed_path / (doc_id.replace("/", "__") + ".json")
        out_file.write_text(json.dumps(doc_data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    print(f"Ingested {count} documents into '{processed_dir}/' (excluded {skipped}).")


if __name__ == "__main__":
    ingest_corpus()