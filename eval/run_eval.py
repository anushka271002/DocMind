"""
eval/run_eval.py
-----------------
Runs every question in eval/eval_set.json through the FULL pipeline
(hybrid retrieval -> rerank -> generation) and reports:
  - Retrieval precision/recall: did the chunks we actually fed the LLM include
    the chunk the answer should have come from?
  - Answer correctness: does the generated answer contain the expected answer?
  - Abstention accuracy: for "not_answerable" questions, did the system correctly
    say "I don't know" instead of hallucinating?

WHY these three metrics specifically (and why they're reported separately, not
merged into one score):
  Retrieval and generation can fail INDEPENDENTLY, and conflating them into one
  number hides which stage to fix. Retrieval recall low + answer correctness low
  -> the problem is retrieval (bad chunking, wrong embedding model, bad rerank).
  Retrieval recall high + answer correctness low -> the problem is the PROMPT or
  the LLM, not retrieval; reranking/chunking are fine. This separation is exactly
  what you'd walk an interviewer through: "here's how I isolated whether my bug
  was in retrieval or generation."

WHY the answer-correctness check is a normalized keyword/substring match, not an
exact string match or a full NLP metric like BLEU/ROUGE:
  BLEU/ROUGE compare n-gram overlap, which is built for machine translation and
  penalizes a correct answer for being phrased differently ("30 VDC" vs "the
  maximum is 30 volts DC") even when it's factually right. For short factual
  answers (numbers, part names, ratings) — which is what expected_answer is
  designed to hold — checking whether the key expected phrase appears in the
  generated answer (after normalizing case/whitespace/punctuation) is simpler,
  more interpretable, and matches what actually matters here: was the right FACT
  present. This is a known limitation for eval_set rows with more open-ended
  expected answers, which is why eval_set.py's guidance is to keep expected_answer
  short and checkable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import answer_question

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"
REPORT_MD_PATH = Path(__file__).resolve().parent / "eval_report.md"
RESULTS_CSV_PATH = Path(__file__).resolve().parent / "eval_results.csv"

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.%/-]", "", text)  # keep chars relevant to units/values (%, /, -, .)
    text = re.sub(r"\s+", " ", text)
    return text


# Common unit spelling variants collapse to one canonical form before comparison, so
# "24 VDC" == "24 V" == "24 volts" and "up to 3 A" == "3A" don't fail a strict-string
# check purely over how the LLM chose to phrase a unit. Order matters: longer/more
# specific patterns first, so e.g. "milliamps" is replaced before a bare "a" pattern
# could accidentally match part of it.
_UNIT_SYNONYMS = [
    (r"\bvolts?\b", "v"),
    (r"\bvdc\b", "v"),
    (r"\bv dc\b", "v"),
    (r"\bmilliamps?\b", "ma"),
    (r"\bmilliampere?s?\b", "ma"),
    (r"\bamps?\b", "a"),
    (r"\bamperes?\b", "a"),
    (r"\bmegahertz\b", "mhz"),
    (r"\bkilohertz\b", "khz"),
    (r"\bdegrees?\s*c\b", "c"),
    (r"\bdeg\s*c\b", "c"),
    (r"\bcelsius\b", "c"),
    (r"\bkilobytes?\b", "kb"),
    (r"\bmegabytes?\b", "mb"),
    (r"\bgigabytes?\b", "gb"),
]


def _canonicalize_units(text: str) -> str:
    """Collapse common unit-spelling variants to one form, and normalize numeric
    ranges ("6 to 21", "6 - 21") to a single "6-21" form, so semantically identical
    values phrased differently don't fail a substring comparison."""
    for pattern, replacement in _UNIT_SYNONYMS:
        text = re.sub(pattern, replacement, text)
    # "6 to 21" -> "6-21"; also collapse spaced hyphens "6 - 21" -> "6-21"
    text = re.sub(r"(\d)\s*(?:to|-)\s*(\d)", r"\1-\2", text)
    text = re.sub(r"\bup to\s+", "", text)  # "up to 3 a" -> "3 a", matches bare "3 a"
    return text


def _strip_citations(text: str) -> str:
    """Remove bracketed citations like '[file.pdf, p.7]' before matching, so a page
    number or filename digit can never accidentally satisfy a numeric expected_answer
    (e.g. expected_answer='7' spuriously matching a citation to page 7)."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def _extract_numbers(text: str) -> list[str]:
    """Pull out standalone numeric tokens (including decimals and negatives) for a
    fallback check: if the exact-phrase match fails, but every number in the expected
    answer appears somewhere in the generated answer, that's strong evidence the
    answer is correct, just phrased differently (different unit spelling, reordered
    Min/Max, extra explanatory text, etc.)."""
    return re.findall(r"-?\d+\.?\d*", text)


# normalized once, at import time, through the SAME _normalize() used on generated
# answers -- avoids the class of bug where the raw phrase contains characters (like
# apostrophes) that _normalize() strips from one side of the comparison but not the other
_ABSTAIN_PHRASES = tuple(_normalize(p) for p in ("I don't know", "don't know based on"))


def _chunk_matches_expected(chunk: dict, row: dict) -> bool:
    if not row.get("expected_source_file"):
        return False
    same_file = chunk["source_file"] == row["expected_source_file"]
    expected_page = row.get("expected_page")
    same_page = (expected_page is None) or (chunk["page_number"] == expected_page)
    return same_file and same_page


def retrieval_metrics(retrieved_chunks: list[dict], row: dict) -> tuple[float, float]:
    """
    Precision@k and recall@k for a single question, where k = len(retrieved_chunks).
    Since each question in this eval set has exactly one "correct" source
    chunk/page (a simplification -- fine for single-fact lookups, less precise
    for multi-chunk synthesis questions), recall is binary (found it or not) and
    precision is (1 if found else 0) / k.
    """
    hits = sum(1 for c in retrieved_chunks if _chunk_matches_expected(c, row))
    precision = hits / len(retrieved_chunks) if retrieved_chunks else 0.0
    recall = 1.0 if hits > 0 else 0.0
    return precision, recall


def answer_correctness(generated_answer: str, row: dict) -> bool:
    """
    Two-tier check, from strictest to most lenient:
    1. Abstention questions: does the answer contain an "I don't know" phrase.
    2. Exact substring: does the normalized expected_answer appear verbatim in the
       normalized generated answer. Fast, precise, but brittle to phrasing.
    3. Lenient fallback: after canonicalizing units/ranges and stripping citation
       brackets, does the exact substring now match? If still no, check that EVERY
       number in expected_answer appears somewhere in the generated answer AND (for
       expected answers containing letters, e.g. part numbers/names) the alphabetic
       content also appears. This catches "24 V" vs "24 VDC", "up to 3 A" vs "3A",
       "6 to 21 volts" vs "6-21 V", without becoming a full NLP similarity metric.
    This tiering exists because a purely strict match seriously undercounted correct
    answers in practice (verified by hand against eval_results.csv rows where the
    generated answer was factually right but phrased with a different unit spelling
    than expected_answer) -- see eval/run_eval.py module docstring for the general
    rationale on why substring matching over BLEU/ROUGE, and this is the fix for
    substring matching being too strict in the OTHER direction.
    """
    norm_answer = _normalize(generated_answer)
    if row["question_type"] == "not_answerable":
        return any(phrase in norm_answer for phrase in _ABSTAIN_PHRASES)

    expected = _normalize(row["expected_answer"])
    if expected in norm_answer:
        return True

    # Lenient fallback
    clean_answer = _canonicalize_units(_strip_citations(norm_answer))
    clean_expected = _canonicalize_units(expected)
    if clean_expected in clean_answer:
        return True

    expected_numbers = _extract_numbers(clean_expected)
    if expected_numbers:
        numbers_present = all(num in clean_answer for num in expected_numbers)
        # also require any alphabetic "core" word (3+ letters, e.g. a unit or part
        # fragment) from expected_answer to appear, so "24" alone can't match an
        # unrelated answer that happens to also contain the digit 24 somewhere
        alpha_words = [w for w in re.findall(r"[a-z]{3,}", clean_expected)]
        alpha_ok = all(w in clean_answer for w in alpha_words) if alpha_words else True
        if numbers_present and alpha_ok:
            return True

    return False


def run_eval() -> pd.DataFrame:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SET_PATH} not found. Run `python eval/eval_set.py --template` "
            f"and fill it in first."
        )
    with open(EVAL_SET_PATH) as f:
        eval_rows = json.load(f)
    eval_rows = [r for r in eval_rows if r.get("question")]  # skip unfilled placeholders

    if not eval_rows:
        raise ValueError(
            f"{EVAL_SET_PATH} has no filled-in questions. See eval/eval_set.py for the schema."
        )

    records = []
    for i, row in enumerate(eval_rows, start=1):
        print(f"[{i}/{len(eval_rows)}] {row['question']}")
        result = answer_question(row["question"])
        precision, recall = retrieval_metrics(result["final_chunks"], row)
        correct = answer_correctness(result["answer"], row)

        records.append(
            {
                "question": row["question"],
                "question_type": row["question_type"],
                "expected_answer": row["expected_answer"],
                "generated_answer": result["answer"],
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "answer_correct": correct,
                "total_sec": result["timing"]["total_sec"],
            }
        )

    df = pd.DataFrame(records)
    return df


def summarize(df: pd.DataFrame) -> str:
    overall = {
        "Total questions": len(df),
        "Retrieval precision (avg)": f"{df['retrieval_precision'].mean():.2%}",
        "Retrieval recall (avg)": f"{df['retrieval_recall'].mean():.2%}",
        "Answer correctness": f"{df['answer_correct'].mean():.2%}",
        "Avg latency (sec/question)": f"{df['total_sec'].mean():.2f}",
    }

    not_ans = df[df["question_type"] == "not_answerable"]
    if len(not_ans):
        overall["Correct abstention rate"] = f"{not_ans['answer_correct'].mean():.2%}"

    by_type = (
        df.groupby("question_type")
        .agg(
            n=("question", "count"),
            retrieval_recall=("retrieval_recall", "mean"),
            answer_correct=("answer_correct", "mean"),
        )
        .reset_index()
    )
    by_type["retrieval_recall"] = by_type["retrieval_recall"].map(lambda x: f"{x:.0%}")
    by_type["answer_correct"] = by_type["answer_correct"].map(lambda x: f"{x:.0%}")

    lines = ["# Evaluation Report", ""]
    lines.append("## Overall")
    lines.append(tabulate(list(overall.items()), headers=["Metric", "Value"], tablefmt="github"))
    lines.append("")
    lines.append("## By question type")
    lines.append(tabulate(by_type, headers="keys", tablefmt="github", showindex=False))
    lines.append("")
    lines.append("## Per-question detail")
    detail_cols = ["question", "question_type", "retrieval_recall", "answer_correct", "total_sec"]
    lines.append(tabulate(df[detail_cols], headers="keys", tablefmt="github", showindex=False))
    return "\n".join(lines)


def main():
    df = run_eval()
    df.to_csv(RESULTS_CSV_PATH, index=False)

    report = summarize(df)
    with open(REPORT_MD_PATH, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nSaved detailed CSV to {RESULTS_CSV_PATH}")
    print(f"Saved markdown report to {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()