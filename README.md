# Technical Docs RAG — Hybrid Retrieval-Augmented Generation

A hybrid RAG system for querying engineering documentation (datasheets, standards,
manuals) in PDF form. Combines dense (FAISS) + sparse (BM25) retrieval, cross-encoder
reranking, and a citation-enforcing, hallucination-resistant generation prompt —
built to be explainable end-to-end in an interview, not just runnable.

## What it does

You point it at a folder of PDFs. It:
1. Parses each PDF respecting its structure (headings, sections, tables) instead of
   blindly splitting every N characters
2. Indexes the chunks two ways: semantically (embeddings in FAISS) and lexically
   (BM25 keyword search)
3. On a query, retrieves candidates from both, fuses the rankings, re-scores the
   top candidates with a cross-encoder, and asks an LLM to answer **using only
   those chunks** — citing the source file and page, or saying "I don't know" if
   the chunks don't actually answer the question
4. Ships with an evaluation harness that scores retrieval accuracy and answer
   correctness against a labeled question set, and a Streamlit UI to query
   interactively with full transparency into what was retrieved and why


## Why these design choices

| Choice | Why |
|---|---|
| Structure-aware chunking, not fixed-size | Fixed-size splitting cuts sentences and table rows in half, producing chunks that embed to garbled or meaningless vectors. This system detects headings via font-size/bold layout signals and keeps tables atomic. |
| Hybrid (dense + BM25), not dense-only | Dense embeddings cluster by meaning and can under-rank exact part numbers/codes (e.g. confusing "XR-450" with "XR-455"). BM25 nails exact tokens but misses paraphrase. Combining both covers both failure modes. |
| Reciprocal Rank Fusion, not weighted score sum | BM25 and cosine-similarity scores are on incompatible, un-normalized scales. RRF only uses each result's *rank*, so it needs no per-corpus tuning. |
| Cross-encoder reranking as a second stage | Bi-encoders (FAISS/BM25) score query and document independently, which is fast but lossy. Cross-encoders jointly attend over the pair for much better relevance judgments — but are too slow to run over the whole corpus, hence the two-stage "retrieve cheap, rerank precisely" pattern. |
| Citations required + explicit "I don't know" instruction | Numbering and pre-labeling chunks with `[file, page]` before generation lets the model cite inline as it writes, rather than us guessing citations after the fact. Temperature 0 + explicit abstention instruction reduces (does not eliminate) hallucination. |
| Retrieval and answer-correctness metrics reported separately | They can fail independently. Low retrieval recall + low answer correctness → fix retrieval. High recall + low correctness → fix the prompt/LLM, not retrieval. |

## Evaluation results

Evaluated on a 44-question labeled test set spanning 4 Arduino datasheets (UNO SPE Shield,
Nano ESP32, UNO Q, UNO R3), including deliberate cross-document disambiguation questions
and "not answerable" trap questions to test abstention behavior.

| Metric                      | Value  |
|------------------------------|--------|
| Total questions              | 44     |
| Retrieval recall (avg)       | 61%    |
| Answer correctness           | 68%    |
| Correct abstention rate      | 100%   |
| Avg latency (sec/question)   | 10.2   |

**By question type:**

| Type           | n  | Retrieval recall | Answer correct |
|----------------|----|-------------------|-----------------|
| numeric        | 10 | 80%               | 80%             |
| table          | 12 | 83%               | 58%             |
| factual        | 17 | 53%               | 59%             |
| not_answerable | 5  | 0% (n/a)          | 100%            |

**Notable findings during evaluation:**
- The system correctly abstained on every single unanswerable question (0% hallucination
  rate on the trap set) — no fabricated answers even when plausible-sounding chunks existed.
- Found and fixed a real cross-document confusion bug: with multiple similar boards each
  having a "VIN Rating" section, the LLM initially cited the wrong board's chunk when the
  wrong document scored a coincidental keyword match. Fixed via an explicit product-matching
  instruction in the generation prompt.
- Found a genuine inconsistency in a source PDF itself (conflicting clock-speed values in
  two different sections of the same datasheet) — the system correctly surfaced a real,
  cited value; the eval question was removed as ambiguous rather than papering over it.
- Remaining gaps are concentrated in retrieval recall on a corpus with many structurally
  similar documents (8 total Arduino boards) — a known scaling challenge for hybrid
  retrieval, discussed further in Limitations below.
