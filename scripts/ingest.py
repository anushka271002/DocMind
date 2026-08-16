"""
scripts/ingest.py
------------------
Run this first: reads every PDF in data/pdfs/, chunks it, and writes
data/processed/chunks.jsonl.

Usage:
    python scripts/ingest.py
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PDF_DIR, CHUNKS_PATH
from src.ingestion import ingest_folder, save_chunks


def main():
    print(f"Reading PDFs from: {PDF_DIR}")
    chunks = ingest_folder(PDF_DIR)

    save_chunks(chunks, CHUNKS_PATH)

    # --- Summary report, useful sanity check before moving to indexing ---
    n_text = sum(1 for c in chunks if c.chunk_type == "text")
    n_table = sum(1 for c in chunks if c.chunk_type == "table")
    by_file = Counter(c.source_file for c in chunks)
    lengths = [len(c.text) for c in chunks]

    print("\n--- Ingestion summary ---")
    print(f"Total chunks:        {len(chunks)}")
    print(f"  text chunks:       {n_text}")
    print(f"  table chunks:      {n_table}")
    print(f"Avg chunk length:    {sum(lengths) / len(lengths):.0f} chars" if lengths else "n/a")
    print(f"Chunks per document:")
    for fname, count in by_file.items():
        print(f"  {fname}: {count}")
    print(f"\nSaved to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
