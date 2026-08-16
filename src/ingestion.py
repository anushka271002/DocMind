"""
src/ingestion.py
-----------------
Turns a folder of PDFs into a list of retrieval-ready chunks with metadata.

WHY structure-aware chunking instead of naive fixed-size splitting:
  Fixed-size splitting (e.g. "every 500 characters") is the default in most RAG
  tutorials, but for technical documentation it actively hurts retrieval quality:
    - It can cut a sentence in half, so the chunk no longer embeds to a coherent
      meaning ("the maximum operating temperature is" / "150C, exceeding this...").
    - It can slice a table row apart from its header row, so a retrieved chunk like
      "3.3V | 500mA | -40 to 85C" is meaningless without knowing which column is which.
    - It ignores document structure, so a chunk might straddle two unrelated
      sections (end of "Installation" + start of "Troubleshooting"), diluting the
      embedding and confusing the LLM about what section the answer came from.
  This module instead:
    1. Detects headings using font-size/bold signals from the PDF's layout (not just
       text patterns), and groups text under the section it belongs to.
    2. Detects tables separately (via pdfplumber, which is purpose-built for table
       geometry) and keeps each table as a single, unsplit chunk.
    3. Splits remaining prose on sentence boundaries only, so no chunk ever ends
       mid-sentence.
    4. Attaches metadata (source file, page number, section heading) to every chunk,
       which is what lets the final answer cite "Datasheet.pdf, page 4, Electrical
       Characteristics" instead of a vague reference.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import pdfplumber
import regex as re
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHUNK_TARGET_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_SENTENCES


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int          # 1-indexed, human-friendly for citations
    section: str
    chunk_type: str           # "text" or "table"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
# A conservative sentence-boundary regex: splits on '.', '!', '?' followed by
# whitespace + a capital letter/quote, but tries to avoid breaking on common
# abbreviations (e.g., i.e., Fig., No., etc.) which are extremely common in
# engineering docs ("...per Fig. 3." should not become two sentences).
# NOTE: the abbreviations include their trailing period. The split point sits
# immediately after the period, so the negative lookbehind must match text that
# ENDS at that exact position -- "\bfig" (no period) would never match right
# after a "." character, so a version without periods here silently never fires.
# (This bug shipped once already and was caught by tests/test_ingestion.py --
# a good example of why chunking needs dedicated unit tests, not just eyeballing.)
_ABBREVIATIONS = r"(?:e\.g\.|i\.e\.|etc\.|fig\.|figs\.|no\.|vol\.|pp\.|approx\.|max\.|min\.|ref\.|eq\.)"
_SENTENCE_SPLIT_RE = re.compile(
    rf"(?<!{_ABBREVIATIONS})(?<=[.!?])\s+(?=[A-Z0-9\"'])",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without cutting an engineering doc's abbreviations."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
def _extract_blocks_with_style(page: fitz.Page) -> list[dict]:
    """
    Pull text spans with their font size/bold flag AND bounding box from a page.
    We need style info (rather than plain page.get_text('text')) because heading
    detection requires layout signals — a heading LOOKS different (bigger/bolder),
    it doesn't necessarily follow a text pattern we could regex for. We need the
    bbox so we can later tell which lines fall inside a detected table region
    (see _filter_table_region_lines) and skip them, instead of guessing from text.

    IMPORTANT: bold is tracked as a FRACTION of the line's characters, not a
    single True/False flag. Datasheets often bold just a key value INSIDE an
    otherwise normal sentence for emphasis (e.g. "The recommended range is
    **6-21 V**."), which is not a heading. An earlier version of this function
    set is_bold=True if ANY span in the line was bold, which caused exactly
    that kind of sentence -- short, and technically "containing bold text" --
    to be misclassified as a section heading and stripped out of the
    retrievable chunk text entirely (caught via a real retrieval failure: a
    query about VIN voltage came back "I don't know" because the one sentence
    with the actual number had been swallowed as a heading label instead of
    chunk content). Requiring most of the line to be bold avoids this.
    """
    raw = page.get_text("dict")
    spans = []
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            line_text_parts = []
            max_size = 0.0
            bold_chars = 0
            total_chars = 0
            for span in line.get("spans", []):
                t = span.get("text", "")
                if not t.strip():
                    continue
                line_text_parts.append(t)
                max_size = max(max_size, span.get("size", 0))
                total_chars += len(t)
                if "bold" in span.get("font", "").lower():
                    bold_chars += len(t)
            line_text = "".join(line_text_parts).strip()
            is_mostly_bold = total_chars > 0 and (bold_chars / total_chars) >= 0.8
            if line_text:
                spans.append({"text": line_text, "size": max_size, "bold": is_mostly_bold, "bbox": line.get("bbox")})
    return spans


def _is_heading(line: dict, body_font_size: float) -> bool:
    """
    Heuristic: a line is a heading if it's noticeably larger than typical body
    text, OR bold and short. Short + bold + not ending in '.' also catches
    numbered section titles like "4.2 Electrical Characteristics".
    We deliberately keep this heuristic simple and documented, rather than a
    black-box ML layout model, so it's easy to explain and debug in an interview.
    """
    text = line["text"]
    if len(text) > 120:  # headings are short; long lines are prose even if bold
        return False
    size_is_bigger = line["size"] >= body_font_size * 1.15
    looks_like_heading_text = bool(re.match(r"^\d+(\.\d+)*\s+\S", text))  # "4.2 Foo"
    if size_is_bigger:
        return True
    if line["bold"] and (looks_like_heading_text or len(text.split()) <= 8):
        return True
    return False


def _estimate_body_font_size(doc: fitz.Document, sample_pages: int = 5) -> float:
    """Median font size across the first few pages, used as the 'body text' baseline."""
    sizes = []
    for page in doc[: min(sample_pages, len(doc))]:
        for line in _extract_blocks_with_style(page):
            sizes.append(line["size"])
    return statistics.median(sizes) if sizes else 10.0


# ---------------------------------------------------------------------------
# Table extraction (kept as atomic, unsplit chunks)
# ---------------------------------------------------------------------------
def _extract_tables_by_page(pdf_path: Path) -> dict[int, list[dict]]:
    """
    Returns {page_number (1-indexed): [{"markdown": str, "bbox": (x0,top,x1,bottom)}, ...]}.
    pdfplumber is used here specifically because it detects table geometry
    (ruled lines / column alignment) far more reliably than PyMuPDF's plain
    text extraction, which just gives you a flat stream of words.

    We keep the bbox (not just the markdown) for two later steps:
      1. Filtering out raw PyMuPDF text lines that fall inside this region, so the
         same table doesn't ALSO get emitted as garbled prose (the original bug —
         a text-based heuristic can't reliably tell "table row" from "short line").
      2. Determining which section heading a table belongs to, by comparing its
         vertical position against heading positions on the same page.
    """
    tables_by_page: dict[int, list[dict]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                found = page.find_tables()
            except Exception:
                found = []
            if not found:
                continue
            entries = []
            for t in found:
                rows = t.extract()
                if not rows or len(rows) < 2:
                    continue  # skip noise (single-row "tables" from stray ruled lines)
                entries.append({"markdown": _table_to_markdown(rows), "bbox": t.bbox})
            if entries:
                tables_by_page[i] = entries
    return tables_by_page


def _bbox_overlaps(line_bbox, table_bbox, tolerance: float = 2.0) -> bool:
    """True if a text line's bbox falls (mostly) inside a table's bbox, meaning it's
    part of that table and should not be duplicated as a separate prose chunk."""
    if not line_bbox or not table_bbox:
        return False
    lx0, ly0, lx1, ly1 = line_bbox
    tx0, ty0, tx1, ty1 = table_bbox
    # PDF coordinate systems: pdfplumber "top" origin can differ slightly from
    # PyMuPDF's; a small tolerance avoids off-by-a-few-points edge misses.
    return (
        lx0 >= tx0 - tolerance
        and lx1 <= tx1 + tolerance
        and ly0 >= ty0 - tolerance
        and ly1 <= ty1 + tolerance
    )


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list of rows) as markdown, so structure survives
    into the embedding text and the LLM prompt (an LLM reads '| Voltage | Current |'
    far better than a flattened string of cell values)."""
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        # pad/truncate rows that don't match header length (common in messy PDFs)
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main per-document pipeline
# ---------------------------------------------------------------------------
def _pack_sentences_into_chunks(
    sentences: list[str], target_chars: int, min_chars: int, overlap: int
) -> list[str]:
    """
    Greedily pack sentences into ~target_chars chunks, never splitting a sentence.
    Carries the last `overlap` sentence(s) of a chunk into the start of the next,
    so a fact split across a chunk boundary (e.g. a spec value stated in one
    sentence, its condition in the next) isn't orphaned from its neighbor.
    """
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > target_chars and current:
            chunks.append(" ".join(current))
            # start next chunk with overlap from the end of this one
            current = current[-overlap:] if overlap else []
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        tail = " ".join(current)
        # merge tiny trailing fragment into previous chunk rather than keeping
        # a near-empty chunk that would dilute retrieval (a 40-char chunk rarely
        # carries enough signal to be usefully embedded on its own)
        if chunks and len(tail) < min_chars:
            chunks[-1] = chunks[-1] + " " + tail
        else:
            chunks.append(tail)
    return chunks


def process_pdf(pdf_path: Path) -> list[Chunk]:
    """Full pipeline for a single PDF: layout parse -> headings -> tables -> chunks."""
    doc = fitz.open(str(pdf_path))
    body_font_size = _estimate_body_font_size(doc)
    tables_by_page = _extract_tables_by_page(pdf_path)

    chunks: list[Chunk] = []
    current_section = "Introduction"  # default section before any heading is seen
    chunk_counter = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        lines = _extract_blocks_with_style(page)
        page_tables = tables_by_page.get(page_number, [])

        # Drop any text line that geometrically falls inside a detected table region.
        # This replaces a text-pattern guess with an actual layout check, which is
        # what fixed the duplicate "Supply Voltage 9 24 30 VDC" prose chunk bug.
        filtered_lines = [
            line
            for line in lines
            if not any(_bbox_overlaps(line["bbox"], t["bbox"]) for t in page_tables)
        ]

        # Build a single ordered stream of "events" (heading | prose-line | table) by
        # vertical position (top y-coordinate), so a table is attributed to whichever
        # section heading actually precedes it on the page -- fixes the earlier bug
        # where every table was tagged with whatever section was active at the START
        # of the page, regardless of where it actually appeared.
        events = []
        for line in filtered_lines:
            y = line["bbox"][1] if line["bbox"] else 0
            kind = "heading" if _is_heading(line, body_font_size) else "prose"
            events.append((y, kind, line["text"]))
        for t in page_tables:
            y = t["bbox"][1]
            events.append((y, "table", t["markdown"]))
        events.sort(key=lambda e: e[0])

        prose_buffer: list[str] = []

        def flush_prose():
            nonlocal chunk_counter
            text = " ".join(prose_buffer).strip()
            prose_buffer.clear()
            if not text:
                return
            sentences = split_sentences(text)
            for piece in _pack_sentences_into_chunks(
                sentences, CHUNK_TARGET_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP_SENTENCES
            ):
                chunk_counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{pdf_path.stem}_p{page_number}_c{chunk_counter}",
                        text=piece,
                        source_file=pdf_path.name,
                        page_number=page_number,
                        section=current_section,
                        chunk_type="text",
                    )
                )

        for y, kind, text in events:
            if kind == "heading":
                flush_prose()  # section boundary: never let a chunk span two sections
                current_section = text
            elif kind == "prose":
                prose_buffer.append(text)
            elif kind == "table":
                flush_prose()  # keep table as a clean boundary too, not spliced into prose
                chunk_counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{pdf_path.stem}_p{page_number}_t{chunk_counter}",
                        text=text,
                        source_file=pdf_path.name,
                        page_number=page_number,
                        section=current_section,
                        chunk_type="table",
                    )
                )

        flush_prose()

    doc.close()
    return chunks


# ---------------------------------------------------------------------------
# Folder-level entry point
# ---------------------------------------------------------------------------
def ingest_folder(pdf_dir: Path) -> list[Chunk]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Add some PDF files there first "
            f"(see README 'Sample Data' section for public sources)."
        )
    all_chunks: list[Chunk] = []
    for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
        try:
            chunks = process_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [WARN] Failed to process {pdf_path.name}: {e}")
    return all_chunks


def save_chunks(chunks: Iterable[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks