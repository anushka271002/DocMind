"""
config.py
---------
Single source of truth for paths, model names, and tunable pipeline knobs.

WHY a central config instead of hardcoding values in each module:
- In interviews you'll be asked "how would you tune retrieval quality?" — this file
  is where you point. Chunk size, top-k, rerank depth are all here, not buried in code.
- Swapping models (e.g. MiniLM -> bge-base, or Groq -> OpenAI) becomes a one-line change.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file for secrets (API keys) — keeps them out of git

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "data" / "pdfs"                 # drop input PDFs here
PROCESSED_DIR = ROOT_DIR / "data" / "processed"      # chunks.jsonl lives here
INDEX_DIR = ROOT_DIR / "data" / "index"              # FAISS index + BM25 pickle live here
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
CHUNK_META_PATH = INDEX_DIR / "chunk_meta.pkl"        # id -> chunk metadata, aligned to both indexes

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Target size in characters (not tokens, for simplicity/portability). ~600-900 chars
# is roughly 100-180 tokens -- small enough for precise retrieval, large enough to
# keep a coherent idea/paragraph together. Tables are NEVER split regardless of size
# (see src/ingestion.py) because a partial table row is worse than a slightly oversized chunk.
CHUNK_TARGET_CHARS = 800
CHUNK_MIN_CHARS = 200          # merge tiny trailing fragments into the previous chunk
CHUNK_OVERLAP_SENTENCES = 1    # carry the last sentence of a chunk into the next, so
                                # context isn't lost right at a chunk boundary

# ---------------------------------------------------------------------------
# Retrieval (hybrid)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small, fast, good enough for
                                                               # technical prose; swap for
                                                               # "BAAI/bge-base-en-v1.5" for
                                                               # higher quality at more compute cost
DENSE_TOP_K = 15          # candidates pulled from FAISS
SPARSE_TOP_K = 15         # candidates pulled from BM25
HYBRID_TOP_K = 25         # size of the merged candidate pool handed to the reranker
# Reciprocal Rank Fusion constant. Standard default from the original RRF paper;
# not sensitive to tuning, which is exactly why RRF is preferred over trying to
# normalize and weight BM25 scores vs cosine similarity by hand (different scales/distributions).
RRF_K = 60

# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------
RERANKER_MODEL = "BAAI/bge-reranker-base"   # cross-encoder: scores (query, chunk) pairs jointly,
                                             # far more accurate than bi-encoder cosine similarity
                                             # because it can attend query<->chunk tokens directly.
                                             # Too slow to run over the whole corpus (hence hybrid
                                             # retrieval narrows candidates first).
RERANK_TOP_N = 5          # final number of chunks handed to the LLM

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"   # fast + free-tier friendly; swap to a larger Groq model
                                        # or OpenAI by editing src/generation.py's call site
GENERATION_TEMPERATURE = 0.0           # deterministic, factual answers — this is a Q&A tool over
                                        # source documents, not a creative-writing task
MAX_ANSWER_TOKENS = 500

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42