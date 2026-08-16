"""
src/pipeline.py
----------------
Wires ingestion-time-built indexes into a single `answer_question()` call:
  hybrid retrieval (top ~20) -> cross-encoder rerank (top ~4) -> LLM generation.

WHY a separate pipeline module instead of duplicating this logic in app.py and
eval/run_eval.py:
  Both the Streamlit app and the evaluation script need the EXACT same
  retrieval->rerank->generation path. If they diverged even slightly (e.g. app.py
  uses top_n=4 but eval uses top_n=5), your evaluation numbers would no longer
  describe what the app actually does -- which defeats the point of evaluating it.
  One shared function is the single source of truth for "what does querying this
  system actually do."
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import HYBRID_TOP_K, RERANK_TOP_N
from src.retrieval import HybridIndex
from src.reranker import rerank
from src.generation import generate_answer

_index_singleton: HybridIndex | None = None


def get_index() -> HybridIndex:
    """Lazy-loaded singleton so Streamlit doesn't reload the FAISS/BM25 index and
    re-instantiate the embedding model on every single user interaction."""
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = HybridIndex()
        _index_singleton.load()
    return _index_singleton


def answer_question(query: str, hybrid_top_k: int = HYBRID_TOP_K, rerank_top_n: int = RERANK_TOP_N) -> dict:
    """
    Full pipeline for one question. Returns a dict with timing + intermediate
    results at every stage, so the Streamlit UI and eval script can both show/use
    retrieval-stage results, not just the final answer.
    """
    index = get_index()

    t0 = time.time()
    hybrid_candidates = index.search(query, top_k=hybrid_top_k)
    t1 = time.time()

    reranked = rerank(query, hybrid_candidates, top_n=rerank_top_n)
    t2 = time.time()

    result = generate_answer(query, reranked)
    t3 = time.time()

    return {
        "query": query,
        "answer": result["answer"],
        "final_chunks": reranked,               # the ~4 chunks actually shown to the LLM
        "hybrid_candidates": hybrid_candidates,  # full ~20 candidate pool, for debugging/transparency
        "timing": {
            "retrieval_sec": round(t1 - t0, 3),
            "rerank_sec": round(t2 - t1, 3),
            "generation_sec": round(t3 - t2, 3),
            "total_sec": round(t3 - t0, 3),
        },
    }
