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

## Architecture

```
                          ┌─────────────────────┐
                          │   PDFs (data/pdfs)   │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────────────────────┐
                    │   1. INGESTION (src/ingestion.py) │
                    │   - PyMuPDF: text + font/layout    │
                    │   - pdfplumber: table geometry      │
                    │   - heading detection -> sections   │
                    │   - sentence-safe chunking          │
                    │   - tables kept atomic (never split)│
                    └────────────────┬─────────────────┘
                                     │
                                     ▼
                     data/processed/chunks.jsonl
                (text + source_file + page + section + type)
                                     │
                    ┌────────────────────────────────┐
                    │  2. INDEX BUILD (src/retrieval.py)│
                    │                                    │
                    │   ┌──────────────┐ ┌─────────────┐│
                    │   │ Dense (FAISS)│ │ Sparse (BM25)││
                    │   │ MiniLM-L6-v2 │ │  rank_bm25   ││
                    │   └──────────────┘ └─────────────┘│
                    └────────────────────────────────────┘
                                     │
                          saved to data/index/
                                     │
        ┌────────────────────────────────────────────────────┐
        │                  QUERY TIME                          │
        │                                                        │
        │   user question                                        │
        │        │                                                │
        │        ▼                                                │
        │   ┌─────────────────────────────┐                       │
        │   │ Hybrid retrieval (top ~20)    │                       │
        │   │ dense top-15 + sparse top-15  │                       │
        │   │ fused via Reciprocal Rank      │                       │
        │   │ Fusion (RRF)                   │                       │
        │   └───────────────┬───────────────┘                       │
        │                   ▼                                        │
        │   ┌─────────────────────────────┐                          │
        │   │ Cross-encoder rerank          │  src/reranker.py         │
        │   │ (bge-reranker-base)            │                          │
        │   │ 20 candidates -> top 3-5       │                          │
        │   └───────────────┬───────────────┘                          │
        │                   ▼                                           │
        │   ┌─────────────────────────────┐                             │
        │   │ Generation (Groq / Llama 3.1) │  src/generation.py          │
        │   │ - numbered, cited excerpts     │                             │
        │   │ - "I don't know" if ungrounded │                             │
        │   └───────────────┬───────────────┘                             │
        │                   ▼                                              │
        │        Answer + citations + used chunks                          │
        └────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                    ▼
         streamlit run app.py                  python eval/run_eval.py
         (interactive UI)                      (batch scoring + report)
```

## Why these design choices (interview cheat-sheet)

| Choice | Why |
|---|---|
| Structure-aware chunking, not fixed-size | Fixed-size splitting cuts sentences and table rows in half, producing chunks that embed to garbled or meaningless vectors. This system detects headings via font-size/bold layout signals and keeps tables atomic. |
| Hybrid (dense + BM25), not dense-only | Dense embeddings cluster by meaning and can under-rank exact part numbers/codes (e.g. confusing "XR-450" with "XR-455"). BM25 nails exact tokens but misses paraphrase. Combining both covers both failure modes. |
| Reciprocal Rank Fusion, not weighted score sum | BM25 and cosine-similarity scores are on incompatible, un-normalized scales. RRF only uses each result's *rank*, so it needs no per-corpus tuning. |
| Cross-encoder reranking as a second stage | Bi-encoders (FAISS/BM25) score query and document independently, which is fast but lossy. Cross-encoders jointly attend over the pair for much better relevance judgments — but are too slow to run over the whole corpus, hence the two-stage "retrieve cheap, rerank precisely" pattern. |
| Citations required + explicit "I don't know" instruction | Numbering and pre-labeling chunks with `[file, page]` before generation lets the model cite inline as it writes, rather than us guessing citations after the fact. Temperature 0 + explicit abstention instruction reduces (does not eliminate) hallucination. |
| Retrieval and answer-correctness metrics reported separately | They can fail independently. Low retrieval recall + low answer correctness → fix retrieval. High recall + low correctness → fix the prompt/LLM, not retrieval. |

## Project structure

```
rag-tech-docs/
├── config.py                # all paths, model names, tunable knobs — start here
├── requirements.txt
├── .env.example              # copy to .env, add your Groq API key
├── src/
│   ├── ingestion.py          # PDF -> structure-aware chunks
│   ├── retrieval.py          # HybridIndex: FAISS + BM25 + RRF fusion
│   ├── reranker.py           # cross-encoder reranking
│   ├── generation.py         # Groq LLM call, citation-enforcing prompt
│   └── pipeline.py           # ties retrieval -> rerank -> generation together
├── scripts/
│   ├── ingest.py              # CLI: PDFs -> data/processed/chunks.jsonl
│   └── build_index.py         # CLI: chunks.jsonl -> data/index/ (FAISS + BM25)
├── eval/
│   ├── eval_set.py            # schema + template generator for labeled Q&A pairs
│   ├── eval_set.json          # your labeled questions (generate with eval_set.py)
│   └── run_eval.py            # runs eval set through full pipeline, scores + reports
├── app.py                     # Streamlit UI
└── data/
    ├── pdfs/                  # <- put your input PDFs here
    ├── processed/             # chunks.jsonl (generated)
    └── index/                 # FAISS index + BM25 pickle (generated)
```

## How to run it locally

### 1. Setup

```bash
git clone <this-repo>
cd rag-tech-docs
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your free Groq API key from https://console.groq.com/keys
```

### 2. Add documents

Drop PDF files into `data/pdfs/`. See "Sample data" below if you don't have any yet.

### 3. Ingest and index

```bash
python scripts/ingest.py        # PDFs -> chunks.jsonl (prints a chunking summary)
python scripts/build_index.py   # chunks.jsonl -> FAISS + BM25 indexes
```

### 4. Query it

```bash
streamlit run app.py
```
Open the local URL Streamlit prints. Ask a question; expand "Sources used" to see
exactly which chunks the answer came from, and the debug panel to see the full
pre-rerank candidate pool with dense/sparse/RRF scores.

### 5. Evaluate it

```bash
python eval/eval_set.py --template --n 35   # scaffold eval/eval_set.json
# ...fill in the blank rows using questions grounded in YOUR ingested documents...
python eval/run_eval.py                      # runs the full pipeline over every question
```
This writes `eval/eval_report.md` (a markdown table — screenshot this for your
portfolio) and `eval/eval_results.csv` (per-question detail).

## Evaluation results

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
```

## Sample data (public, legal to download)

You'll need real technical PDFs to build a meaningful eval set. A few good sources:

1. **NASA Technical Reports Server (NTRS)** — [ntrs.nasa.gov](https://ntrs.nasa.gov/) —
   hundreds of thousands of public-domain (no copyright) aerospace/engineering
   technical reports, many with detailed spec tables. Search for a narrow topic
   (e.g. "pressure sensor calibration") to get a manageable, topically-coherent set.
2. **NIST Engineering Standards & Publications** — [nvlpubs.nist.gov](https://nvlpubs.nist.gov/) —
   public-domain U.S. government engineering/measurement standards and technical
   notes, often with tables of tolerances/specifications — good for testing
   table-chunking specifically.
3. **Open hardware documentation** — datasheets for widely-used open-source hardware
   platforms (e.g. Arduino, Raspberry Pi, ESP32/Espressif) are freely published by
   the manufacturers for exactly this kind of reuse. Good for realistic electrical-
   characteristics tables and part-number-heavy retrieval tests (e.g. "ESP32 vs
   ESP32-S3" is a great analog to the XR-450/XR-455 disambiguation test in this repo).

Aim for 3-8 documents in the same domain (all sensor datasheets, or all from one
standards body) so your eval questions can meaningfully probe cross-document
retrieval, not just single-document lookup.

## Known limitations (worth mentioning proactively in an interview)

- **Heading detection is heuristic** (font-size/bold based), not a trained layout
  model — works well on typical datasheet/manual formatting, less reliably on
  PDFs with unconventional or purely-image-based layouts (scanned docs need OCR
  first, which this project doesn't include).
- **Answer-correctness scoring is substring/keyword matching**, not semantic —
  intentional trade-off for interpretability over metrics like BLEU/ROUGE, but it
  means `expected_answer` values need to be short and specific (see `eval/eval_set.py`).
- **Retrieval eval assumes one correct source chunk per question** — fine for
  factual/numeric lookups, an oversimplification for questions that legitimately
  need synthesis across multiple chunks.
- **FAISS uses a flat (exact) index**, which is fine at portfolio scale (thousands
  of chunks) but would need an approximate index (IVF/HNSW) at much larger scale.
