"""
scripts/build_index.py
-----------------------
Run this after scripts/ingest.py: loads chunks.jsonl, builds the FAISS dense
index and BM25 sparse index, and saves both to data/index/.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CHUNKS_PATH
from src.ingestion import load_chunks
from src.retrieval import HybridIndex


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `python scripts/ingest.py` first."
        )
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    index = HybridIndex()
    index.build(chunks)
    index.save()
    print("Saved FAISS index, BM25 index, and chunk metadata to data/index/")


if __name__ == "__main__":
    main()
