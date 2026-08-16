import sys
sys.path.insert(0, '.')
from src.ingestion import load_chunks
from src.retrieval import _tokenize_for_bm25
from rank_bm25 import BM25Okapi
from config import CHUNKS_PATH
import numpy as np

chunks = load_chunks(CHUNKS_PATH)
texts = [c['text'] for c in chunks]
tokenized = [_tokenize_for_bm25(t) for t in texts]
bm25 = BM25Okapi(tokenized)

query = "What is the recommended VIN input voltage range for the Nano ESP32?"
scores = bm25.get_scores(_tokenize_for_bm25(query))
ranked_idx = np.argsort(scores)[::-1]

print("Total chunks in corpus:", len(chunks))
print()
print("=== Top 10 BM25 results across ALL documents ===")
for rank, idx in enumerate(ranked_idx[:10], 1):
    c = chunks[idx]
    print(f"{rank}. score={scores[idx]:.2f} [{c['chunk_type']}] {c['source_file']} p{c['page_number']} - {c['section']}")

print()
print("=== Where does the REAL ABX00083 VIN table chunk rank? ===")
for idx, c in enumerate(chunks):
    if c['source_file'] == 'ABX00083-datasheet.pdf' and c['chunk_type'] == 'table' and 'Operating' in c['section']:
        pos = list(ranked_idx).index(idx)
        print(f"Found at BM25 rank #{pos+1} of {len(chunks)}, score={scores[idx]:.2f}")
        print(c['text'])