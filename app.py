"""
app.py
------
Streamlit interface: ask a question, see the generated answer with citations,
and inspect exactly which chunks were retrieved (and by which retriever) for
transparency/debugging.

WHY the retrieved-chunks panel is shown at all, not just the final answer:
  In a portfolio/interview context, showing your work matters more than in a
  pure end-user product. Surfacing which chunks were retrieved, their hybrid
  score, whether dense/sparse/both found them, and their rerank score turns this
  from a black box into something you can screenshot and explain: "here's proof
  the hybrid retriever pulled the right chunk even though the phrasing didn't
  match exactly" or "here's a case where reranking demoted a false positive."

WHY the "view source page" image feature (this pass):
  A citation like "[Datasheet.pdf, p.4]" asks the reader to trust the pipeline.
  Rendering the actual PDF page with the matched text highlighted turns that
  into something checkable at a glance -- you can screenshot "here's the citation,
  and here's proof it's really on that page, not a hallucinated reference."

  IMPLEMENTATION NOTE: Chunk objects (see src/ingestion.py) only store page_number,
  not the exact bounding box of the chunk's text -- bbox is computed during
  ingestion but discarded after chunking. Rather than re-ingesting everything to
  persist bboxes, this searches for the chunk's text on its page at render time
  via PyMuPDF's page.search_for(), which works well for prose chunks since their
  text is a near-verbatim copy of the PDF content. It does NOT work for table
  chunks, because those are stored as reconstructed markdown ("| Voltage | ... |"),
  not the literal PDF text -- so table chunks show the page without a highlight,
  with a caption explaining why, instead of silently failing to highlight anything.

  DEPLOYMENT NOTE: this feature opens the ORIGINAL PDF file from PDF_DIR at
  render time (fitz.open(pdf_path_str)), not just the prebuilt index. Since
  data/pdfs/*.pdf is gitignored (only the built index is committed/deployed),
  the original PDFs will NOT be present in a deployed environment like
  Streamlit Cloud unless you deliberately also commit them. In that case this
  button correctly falls through to the "Original file not found" warning
  below rather than crashing -- that's expected behavior given the current
  deployment setup, not a bug.

DESIGN NOTES (earlier pass):
  Plain, restrained styling -- one accent color, flat pill badges, consistent
  8px-radius cards, light sidebar, standard fonts. Nothing here should call
  attention to itself before the answer does.

Run with: streamlit run app.py
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import GROQ_API_KEY, PDF_DIR
from src.pipeline import answer_question, get_index


st.set_page_config(page_title="DocuMind", page_icon="📘", layout="wide")

# ---------------------------------------------------------------------------
# Minimal styling — one accent color, flat badges, consistent spacing
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --accent: #2452D8;
            --accent-soft: #EAF0FE;
            --text: #1A1D23;
            --muted: #6B7280;
            --border: #E4E6EB;
            --card: #FFFFFF;
            --dense: #2563EB;
            --dense-bg: #EFF4FE;
            --sparse: #92650F;
            --sparse-bg: #FBF2E2;
            --used: #16794F;
            --used-bg: #E9F7F0;
        }

        .block-container { padding-top: 2rem; max-width: 1080px; }

        h1 { font-weight: 650; font-size: 1.55rem; margin-bottom: 0.15rem; }
        .dm-subtitle { color: var(--muted); font-size: 0.92rem; margin-bottom: 1.5rem; max-width: 68ch; }

        /* Chips for timing */
        .dm-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem; }
        .dm-chip {
            font-size: 0.78rem;
            color: var(--muted);
            background: #F5F6F8;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.2rem 0.55rem;
        }

        /* Cards */
        .dm-card {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
            background: var(--card);
        }
        .dm-card-title { font-weight: 600; font-size: 0.92rem; color: var(--text); margin-bottom: 0.2rem; }
        .dm-card-meta { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.4rem; }

        /* Flat badges (no rotation, no gimmick) */
        .dm-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 500;
            border-radius: 999px;
            padding: 0.08rem 0.55rem;
            margin-right: 0.35rem;
        }
        .dm-badge-dense { color: var(--dense); background: var(--dense-bg); }
        .dm-badge-sparse { color: var(--sparse); background: var(--sparse-bg); }
        .dm-badge-used { color: var(--used); background: var(--used-bg); }

        .dm-empty {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.75rem;
            text-align: center;
            color: var(--muted);
            background: #FAFAFA;
        }

        .stTabs [data-baseweb="tab"] { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📘 DocuMind")
st.markdown(
    '<div class="dm-subtitle">Hybrid RAG (dense + BM25 + cross-encoder reranking) over your '
    "engineering PDFs. Answers are grounded ONLY in retrieved excerpts, with citations.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Source-page rendering: find the chunk's text on its page and highlight it
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def render_source_page(pdf_path_str: str, page_number: int, chunk_text: str, chunk_type: str, zoom: float = 1.8):
    """
    Renders one PDF page as a PNG, with the chunk's text highlighted if it can
    be located on the page. Returns (png_bytes, match_found: bool).
    Cached because re-searching + re-rasterizing the same page on every rerun
    (e.g. toggling an unrelated slider) would otherwise reopen the PDF each time.
    """
    doc = fitz.open(pdf_path_str)
    page = doc[page_number - 1]

    match_found = False
    if chunk_type != "table":
        # Exact match first -- chunk text is often a near-verbatim copy of the
        # page. If that fails (whitespace/line-break differences from how spans
        # were joined during chunking), fall back to searching sentence-by-sentence
        # so a highlight still appears even if the whole chunk doesn't match as
        # one string.
        rects = page.search_for(chunk_text)
        if not rects:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk_text):
                sentence = sentence.strip()
                if len(sentence) < 8:  # too short -> matches everywhere, not useful
                    continue
                rects.extend(page.search_for(sentence))
        for r in rects:
            page.add_highlight_annot(r)
        match_found = len(rects) > 0

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes, match_found


def show_source_page_button(chunk: dict, button_key: str):
    """Renders a toggle button that reveals the highlighted source page on click,
    rather than rendering every page eagerly (which would be slow with several
    sources open at once)."""
    state_key = f"show_{button_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    label = "Hide source page" if st.session_state[state_key] else "View source page"
    if st.button(label, key=f"btn_{button_key}"):
        st.session_state[state_key] = not st.session_state[state_key]

    if st.session_state[state_key]:
        pdf_path = PDF_DIR / chunk["source_file"]
        if not pdf_path.exists():
            st.warning(f"Original file not found at `{pdf_path}` — can't render the page image.")
            return
        with st.spinner("Rendering page..."):
            png_bytes, match_found = render_source_page(
                str(pdf_path), chunk["page_number"], chunk["text"], chunk["chunk_type"]
            )
        st.image(png_bytes, caption=f"{chunk['source_file']} — page {chunk['page_number']}", use_container_width=True)
        if chunk["chunk_type"] == "table":
            st.caption(
                "This chunk is a reconstructed table (markdown), not literal PDF text, "
                "so it can't be highlighted on the page — showing the page as-is."
            )
        elif not match_found:
            st.caption("Couldn't locate an exact text match on the page to highlight — showing the page as-is.")


# ---------------------------------------------------------------------------
# Startup checks — fail loudly and helpfully rather than a raw traceback
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    st.error(
        "No GROQ_API_KEY found. Copy `.env.example` to `.env` and add your key "
        "from https://console.groq.com/keys, then restart the app."
    )
    st.stop()

try:
    with st.spinner("Loading index (embedding model, FAISS, BM25)..."):
        index = get_index()
except FileNotFoundError:
    st.error(
        "No index found. Run these two commands first, then restart the app:\n\n"
        "```\npython scripts/ingest.py\npython scripts/build_index.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — corpus + pipeline controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Corpus")
    n_docs = len(set(c["source_file"] for c in index.chunks))
    col_a, col_b = st.columns(2)
    col_a.metric("Chunks", len(index.chunks))
    col_b.metric("Documents", n_docs)
    with st.expander("Source files"):
        for fname in sorted(set(c["source_file"] for c in index.chunks)):
            st.write(f"- {fname}")

    st.divider()
    st.header("Pipeline settings")
    hybrid_top_k = st.slider("Hybrid candidates (before rerank)", 5, 30, 20)
    rerank_top_n = st.slider("Chunks used for the answer (after rerank)", 1, 8, 4)

    st.divider()
    st.caption(
        "🔵 dense = FAISS embedding match · 🟡 sparse = BM25 keyword match · "
        "🟢 used = made it into the final answer after reranking"
    )

# ---------------------------------------------------------------------------
# Query box
# ---------------------------------------------------------------------------
query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. What is the maximum supply voltage for the XR-450?",
)

if query:
    with st.spinner("Retrieving, reranking, and generating..."):
        result = answer_question(query, hybrid_top_k=hybrid_top_k, rerank_top_n=rerank_top_n)

    t = result["timing"]

    tab_answer, tab_sources, tab_debug = st.tabs(
        [
            "Answer",
            f"Sources ({len(result['final_chunks'])})",
            f"Debug pool ({len(result['hybrid_candidates'])})",
        ]
    )

    # --- Answer tab ---
    with tab_answer:
        st.markdown(result["answer"])
        st.markdown(
            f"""
            <div class="dm-chips">
                <span class="dm-chip">retrieval {t['retrieval_sec']}s</span>
                <span class="dm-chip">rerank {t['rerank_sec']}s</span>
                <span class="dm-chip">generation {t['generation_sec']}s</span>
                <span class="dm-chip">total {t['total_sec']}s</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Sources tab: the chunks actually shown to the LLM, each with an
    #     optional highlighted page-image view ---
    with tab_sources:
        for i, c in enumerate(result["final_chunks"], start=1):
            badges = '<span class="dm-badge dm-badge-used">used</span>'
            if c.get("in_dense_top_k"):
                badges += '<span class="dm-badge dm-badge-dense">dense</span>'
            if c.get("in_sparse_top_k"):
                badges += '<span class="dm-badge dm-badge-sparse">sparse</span>'
            st.markdown(
                f"""
                <div class="dm-card">
                    <div class="dm-card-title">[{i}] {c['source_file']} — p.{c['page_number']} — {c['section']}</div>
                    <div class="dm-card-meta">rerank score: {c.get('rerank_score', 0):.3f}</div>
                    {badges}
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Show excerpt text"):
                st.markdown(c["text"])
            show_source_page_button(c, button_key=f"src_{c.get('chunk_id', i)}")
            st.markdown("&nbsp;", unsafe_allow_html=True)  # small spacer before next card

    # --- Debug tab: full hybrid candidate pool before reranking ---
    with tab_debug:
        st.caption("Full candidate pool from hybrid (dense + BM25) retrieval, before cross-encoder reranking.")
        for i, c in enumerate(result["hybrid_candidates"], start=1):
            dense_s = f"{c['dense_score']:.3f}" if c.get("dense_score") is not None else "—"
            sparse_s = f"{c['sparse_score']:.3f}" if c.get("sparse_score") is not None else "—"
            used = c in result["final_chunks"]
            badge = '<span class="dm-badge dm-badge-used">used</span>' if used else ""
            st.markdown(
                f"""
                <div class="dm-card">
                    <div class="dm-card-title">[{i}] {c['source_file']} — p.{c['page_number']}</div>
                    <div class="dm-card-meta">RRF: {c['retrieval_score']:.4f} · dense: {dense_s} · sparse: {sparse_s}</div>
                    {badge}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text(c["text"][:250] + ("..." if len(c["text"]) > 250 else ""))
else:
    st.info("Enter a question above to query your documents.")
    # Check the loaded INDEX for content, not raw PDF files on disk -- in a
    # deployed environment (e.g. Streamlit Cloud), the source PDFs are correctly
    # excluded from the repo once ingested (only the prebuilt index is shipped),
    # so checking for .pdf files here would show a misleading warning even when
    # everything is working correctly. This only guards the initial "nothing
    # ingested yet" empty state -- unrelated to the "View source page" feature's
    # own, separate need for the original PDFs (see its DEPLOYMENT NOTE above).
    if not index.chunks:
        st.warning(
            f"The index has no chunks. If running locally, add PDFs to `{PDF_DIR}` "
            "and run `python scripts/ingest.py && python scripts/build_index.py`. "
            "If deployed, make sure `data/index/` was committed and pushed."
        )