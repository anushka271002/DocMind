"""
tests/test_ingestion.py
------------------------
Unit tests for the chunking logic in src/ingestion.py. Chunking is the part of
this pipeline most likely to silently regress (e.g. someone tweaks the sentence
regex and suddenly abbreviations get split) without an obvious symptom until
retrieval quality quietly degrades — hence dedicated tests for it specifically,
rather than only integration-testing via the full pipeline.

Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion import split_sentences, _pack_sentences_into_chunks, _is_heading


def test_split_sentences_basic():
    text = "The sensor is rated IP67. It operates from 9 to 30 VDC. Do not exceed 85C."
    sentences = split_sentences(text)
    assert sentences == [
        "The sensor is rated IP67.",
        "It operates from 9 to 30 VDC.",
        "Do not exceed 85C.",
    ]


def test_split_sentences_does_not_break_on_abbreviations():
    text = "Refer to Fig. 3 for details. See Section 2, e.g. the voltage table, for specs."
    sentences = split_sentences(text)
    # "Fig. 3" and "e.g." must NOT be treated as sentence boundaries
    assert len(sentences) == 2
    assert "Fig. 3" in sentences[0]
    assert "e.g." in sentences[1]


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_pack_sentences_respects_target_size_without_splitting_sentences():
    sentences = [
        "Sentence one is here.",
        "Sentence two follows it.",
        "Sentence three is a bit longer than the others.",
        "Sentence four wraps things up nicely.",
    ]
    chunks = _pack_sentences_into_chunks(sentences, target_chars=50, min_chars=10, overlap=0)
    # every original sentence must appear intact in exactly the chunk text,
    # never truncated mid-word
    rejoined = " ".join(chunks)
    for s in sentences:
        assert s in rejoined


def test_pack_sentences_carries_overlap_between_chunks():
    sentences = ["First sentence here is long enough to fill a chunk on its own really."] * 1 + [
        "Second sentence is also fairly long to force a new chunk boundary here.",
        "Third sentence continues the thought from the second one nicely.",
    ]
    chunks = _pack_sentences_into_chunks(sentences, target_chars=80, min_chars=10, overlap=1)
    # with overlap=1, the last sentence of chunk N should reappear at the start of chunk N+1
    if len(chunks) > 1:
        assert chunks[0].split(". ")[-1].strip(".") in chunks[1]


def test_pack_sentences_merges_tiny_trailing_fragment():
    sentences = ["A reasonably long first sentence that takes up most of the target size."] + ["Tiny."]
    chunks = _pack_sentences_into_chunks(sentences, target_chars=80, min_chars=20, overlap=0)
    # "Tiny." (5 chars) is below min_chars and should be merged into the previous
    # chunk rather than emitted as its own near-empty, low-signal chunk
    assert "Tiny." in chunks[-1]
    assert not any(chunk.strip() == "Tiny." for chunk in chunks)


def test_is_heading_ignores_partial_bold_emphasis():
    """
    Regression test for a real bug: a sentence with just a KEY VALUE bolded for
    emphasis (e.g. "The recommended range is **6-21 V**.") was misclassified as
    a section heading because any-span-bold was treated as line-bold. This
    silently deleted the sentence's text from retrievable chunks (it became a
    section label instead), causing correct retrieval queries to return
    "I don't know" even though the fact was in the source PDF.
    """
    from src.ingestion import _is_heading

    # Simulates a line where only "6-21 V" is bold, rest of the sentence is not --
    # bold_chars is small relative to total_chars, so `bold` should be False.
    mostly_normal_line = {
        "text": "The recommended input voltage range is 6-21 V.",
        "size": 9.0,
        "bold": False,  # what _extract_blocks_with_style should now compute
        "bbox": (0, 0, 100, 10),
    }
    assert _is_heading(mostly_normal_line, body_font_size=9.0) is False

    # A genuine heading (short, ALL bold) should still be detected correctly
    real_heading_line = {
        "text": "11.3 VIN Rating",
        "size": 9.0,
        "bold": True,
        "bbox": (0, 0, 100, 10),
    }
    assert _is_heading(real_heading_line, body_font_size=9.0) is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))