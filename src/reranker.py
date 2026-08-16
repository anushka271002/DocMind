"""
src/reranker.py
----------------
Re-scores the hybrid retrieval candidates with a cross-encoder and keeps the
best few for generation.

WHY a cross-encoder re-ranker on top of hybrid retrieval, instead of just
trusting the hybrid (RRF) ranking directly:
  Both FAISS dense search and BM25 score a QUERY and a DOCUMENT independently
  and then compare the two fixed representations (bi-encoder architecture).
  That's what makes them fast enough to search thousands of chunks, but it's
  also inherently lossy: the model never actually looks at the query and the
  candidate chunk TOGETHER.
  A cross-encoder reranker instead takes the (query, chunk) pair as joint input
  and lets the model's attention mechanism directly compare query tokens against
  chunk tokens, producing a much more accurate relevance judgment. The catch is
  cost: cross-encoders are too slow to run over an entire corpus (you'd need one
  forward pass per document per query). So the standard pattern -- and the reason
  this pipeline retrieves top-20 with cheap hybrid search FIRST -- is to use the
  cheap retriever to narrow thousands of chunks down to ~20 plausible candidates,
  then spend the expensive, accurate cross-encoder pass only on those 20.
  This two-stage "retrieve cheap, rerank precisely" pattern is standard in
  production search/RAG systems for exactly this cost/accuracy tradeoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RERANKER_MODEL, RERANK_TOP_N

_reranker_instance: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy singleton load -- the cross-encoder is only needed at query time, and
    loading it eagerly at import time would slow down every script that imports
    this module even when reranking isn't used yet (e.g. during ingestion)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Takes the ~15-20 hybrid retrieval candidates and returns the best `top_n`,
    re-scored by the cross-encoder. Each candidate dict is annotated with
    `rerank_score` so the Streamlit app can show it for transparency.
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)  # higher = more relevant

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]
