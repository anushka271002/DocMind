"""
eval/eval_set.py
-----------------
Defines the schema for the evaluation set and provides a template generator.

WHY a labeled eval set at all (this is the answer to "how do you know your RAG
system actually works?" in an interview):
  Without a labeled set, "does the RAG system work" is just a vibe based on a
  handful of manual queries you happened to try. A labeled set lets you measure
  retrieval and generation quality OBJECTIVELY, track regressions when you change
  chunk size / models / prompts, and produce a defensible number for your resume
  ("94% retrieval recall@5, 87% answer correctness") instead of an anecdote.

SCHEMA — each question needs:
  - question: str
  - expected_answer: str          (a short reference answer, used for correctness scoring)
  - expected_source_file: str     (which PDF the answer should come from)
  - expected_page: int | None     (which page -- None if you're not sure/it spans pages)
  - question_type: str            ("factual", "numeric", "table", "not_answerable")
      "not_answerable" questions are IMPORTANT: include a few questions whose answer
      is NOT in your corpus, to verify the "I don't know" behavior actually fires
      instead of hallucinating. A RAG eval set with 100% answerable questions can't
      tell you whether your anti-hallucination prompt actually works.

HOW TO BUILD YOUR 30-40 QUESTIONS:
  1. Run `python scripts/ingest.py` on your real PDFs first, so you can read
     data/processed/chunks.jsonl and write questions grounded in your actual content.
  2. Aim for a MIX:
       - ~40% simple factual lookups ("What is the max operating temperature of the X-200?")
       - ~25% numeric/spec lookups that hit tables ("What is the accuracy rating in Table 2?")
       - ~20% questions needing light synthesis across 2 sentences/chunks
       - ~15% "not_answerable" -- ask about a product/spec NOT in your corpus
  3. Write expected_answer as a short, checkable phrase (a number+unit, a part
     name, a yes/no) -- NOT a full paragraph, since eval/run_eval.py does substring/
     fuzzy matching against it (see eval/run_eval.py's `answer_correctness` for the
     exact matching approach and its limitations).

Below is a small STARTER TEMPLATE (5 example rows) matching the sample datasheet
described in the README. Copy this file's `EXAMPLES` structure to build your real
eval/eval_set.json once you have real documents ingested -- or run
`python eval/eval_set.py --template` to write a JSON skeleton to fill in.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

EVAL_SET_PATH = Path(__file__).resolve().parent / "eval_set.json"

# A few worked examples, written against the synthetic "XR-450" sample datasheet
# used during development (see README's "Sample Data" section for real corpora).
EXAMPLES = [
    {
        "question": "What is the maximum supply voltage for the XR-450?",
        "expected_answer": "30 VDC",
        "expected_source_file": "sample_datasheet.pdf",
        "expected_page": 1,
        "question_type": "numeric",
    },
    {
        "question": "What ingress protection rating does the XR-450 housing have?",
        "expected_answer": "IP67",
        "expected_source_file": "sample_datasheet.pdf",
        "expected_page": 1,
        "question_type": "factual",
    },
    {
        "question": "What is the accuracy specification of the XR-450 at typical conditions?",
        "expected_answer": "0.25% FS",
        "expected_source_file": "sample_datasheet.pdf",
        "expected_page": 1,
        "question_type": "table",
    },
    {
        "question": "What thread type is used to mount the XR-450?",
        "expected_answer": "1/4-18 NPT",
        "expected_source_file": "sample_datasheet.pdf",
        "expected_page": 1,
        "question_type": "factual",
    },
    {
        "question": "What is the wireless Bluetooth range of the XR-450?",
        "expected_answer": "I don't know based on the provided documents.",
        "expected_source_file": None,
        "expected_page": None,
        "question_type": "not_answerable",
    },
]


def write_template(n_placeholders: int = 30) -> None:
    """Writes eval_set.json seeded with the worked examples plus blank placeholders
    for you to fill in against your own corpus."""
    placeholders = [
        {
            "question": "",
            "expected_answer": "",
            "expected_source_file": "",
            "expected_page": None,
            "question_type": "factual",  # factual | numeric | table | not_answerable
        }
        for _ in range(max(0, n_placeholders - len(EXAMPLES)))
    ]
    data = EXAMPLES + placeholders
    with open(EVAL_SET_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(data)} rows ({len(EXAMPLES)} filled examples + {len(placeholders)} "
          f"blank placeholders) to {EVAL_SET_PATH}")
    print("Fill in the blank rows using your own ingested documents, then run eval/run_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", action="store_true", help="Write a fillable eval_set.json")
    parser.add_argument("--n", type=int, default=30, help="Total number of rows to scaffold")
    args = parser.parse_args()
    if args.template:
        write_template(args.n)
    else:
        print("Run with --template to generate eval/eval_set.json")
