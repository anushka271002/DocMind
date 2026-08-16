"""
src/retrieval.py
-----------------
Builds and queries a hybrid (dense + sparse) retrieval index over the chunks
produced by src/ingestion.py.

WHY hybrid retrieval instead of pure dense (embedding-only) search:
  Dense embeddings are great at semantic similarity ("what protects against
  overheating?" matching a passage about thermal shutdown) but they routinely
  under-rank exact tokens that matter a lot in technical documents: part numbers
  ("XR-450"), error codes, units, and precise numeric values. Embedding models are
  trained to cluster texts by MEANING, so "XR-450" and "XR-455" can end up close
  together in vector space even though a user searching for one very specifically
  does NOT want the other.
  BM25 (sparse/keyword search) is the opposite: it excels at exact term matching
  but has no notion of synonyms or paraphrase ("max operating temp" won't match
  "maximum permissible temperature" as well as a dense model would).
  Combining both means a query like "What is the max supply voltage for the
  XR-450?" gets BM25's precision on "XR-450" AND dense retrieval's semantic
  understanding of "max supply voltage" phrased differently than the source text.

WHY Reciprocal Rank Fusion (RRF) to combine them, rather than a weighted score sum:
  BM25 scores and cosine similarity scores live on completely different, un-normalized
  scales (BM25 is unbounded and corpus-dependent; cosine similarity is bounded [-1,1]
  but often bunched in a narrow range for sentence embeddings). Weighting and summing
  raw scores requires manual tuning per corpus and breaks when the corpus changes size.
  RRF instead only looks at each result's RANK in its own list (1st, 2nd, 3rd place),
  which is scale-free and robust with no corpus-specific tuning required:
      RRF_score(doc) = sum over retrievers of  1 / (k + rank_in_that_retriever)
  A doc that ranks well in EITHER retriever gets a strong combined score; a doc that
  ranks well in BOTH gets boosted further.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    BM25_INDEX_PATH,
    CHUNK_META_PATH,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
)


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Lowercase whitespace/punctuation tokenizer for BM25.
    Deliberately simple (no stemming) because technical part numbers and units
    ("XR-450", "IP67", "4-20mA") are exact strings we do NOT want a stemmer
    mangling -- stemming is built for natural-language words, not part numbers.
    """
    import regex as re
    return re.findall(r"[a-z0-9][a-z0-9\-\./]*", text.lower())


class HybridIndex:
    """Owns the dense (FAISS) and sparse (BM25) indexes plus the chunk metadata
    they're aligned to, and exposes a single hybrid `search()` call."""

    def __init__(self):
        self.embedder = None
        self.faiss_index = None
        self.bm25 = None
        self.chunks: list[dict] = []  # aligned by integer position to both indexes

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("No chunks to index. Run scripts/ingest.py first.")
        self.chunks = chunks

        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        texts = [c["text"] for c in chunks]

        # --- Dense: encode all chunks and build a FAISS flat index ---
        # IndexFlatIP (inner product) on L2-normalized vectors == cosine similarity.
        # We use a flat (exhaustive) index rather than an approximate index (e.g. IVF/HNSW)
        # because a portfolio-scale corpus (thousands of chunks) is small enough that
        # exact search is fast and there's no need to trade accuracy for speed -- that
        # trade-off only pays off at much larger scale.
        print("Encoding chunks for dense index...")
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings.astype(np.float32))

        # --- Sparse: build BM25 over tokenized chunks ---
        print("Building BM25 index...")
        tokenized = [_tokenize_for_bm25(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        print(f"Indexed {len(chunks)} chunks.")

    def save(self) -> None:
        FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(CHUNK_META_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> None:
        if not (FAISS_INDEX_PATH.exists() and BM25_INDEX_PATH.exists() and CHUNK_META_PATH.exists()):
            raise FileNotFoundError(
                "Index files not found. Run scripts/build_index.py first."
            )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(CHUNK_META_PATH, "rb") as f:
            self.chunks = pickle.load(f)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.faiss_index.search(q_vec.astype(np.float32), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokenized_query = _tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Returns up to `top_k` chunks (dicts, with retrieval metadata attached),
        ranked by fused RRF score, highest first.
        """
        dense_results = self._dense_search(query, DENSE_TOP_K)
        sparse_results = self._sparse_search(query, SPARSE_TOP_K)

        # --- Reciprocal Rank Fusion ---
        rrf_scores: dict[int, float] = {}
        for rank, (idx, _score) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _score) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        # keep raw scores around too, purely for transparency in the Streamlit UI
        dense_score_by_idx = dict(dense_results)
        sparse_score_by_idx = dict(sparse_results)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for idx, fused_score in ranked:
            chunk = dict(self.chunks[idx])  # copy so we don't mutate the stored index
            chunk["retrieval_score"] = fused_score
            chunk["dense_score"] = dense_score_by_idx.get(idx)
            chunk["sparse_score"] = sparse_score_by_idx.get(idx)
            chunk["in_dense_top_k"] = idx in dense_score_by_idx
            chunk["in_sparse_top_k"] = idx in sparse_score_by_idx
            results.append(chunk)
        return results
