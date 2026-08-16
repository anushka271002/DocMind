"""
src/generation.py
------------------
Takes the final reranked chunks and the user's question, and generates a
grounded answer via the Groq API.

WHY the prompt is built the way it is (this is the anti-hallucination design,
and the part you should be ready to explain in most depth in interviews):
  1. Chunks are numbered and labeled with [source_file, page] BEFORE the answer
     is generated, not after. This lets the model cite inline as it writes
     ("...per [Datasheet.pdf, p.4]...") rather than us trying to guess citations
     after the fact by string-matching the answer against chunks (unreliable).
  2. The system prompt explicitly instructs the model to answer ONLY from the
     provided context and to say "I don't know" if the context doesn't contain
     the answer. LLMs are trained to be helpful, which left unchecked biases them
     toward confidently answering from parametric (training) knowledge even when
     asked to stick to given context -- this instruction, combined with temperature=0
     to reduce answer variance, and NOT passing "system knowledge" framing, all
     push the model toward abstaining rather than confabulating.
  3. We keep the retrieved chunks as ordered, separate blocks (not concatenated
     into one blob) so the model can differentiate "this fact is from page 2, this
     other fact is from page 5" -- concatenation would blur that boundary and make
     accurate per-fact citation harder.
"""

from __future__ import annotations

import sys
from pathlib import Path

from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL, GENERATION_TEMPERATURE, MAX_ANSWER_TOKENS


SYSTEM_PROMPT = """You are a technical documentation assistant. You answer questions \
about engineering/equipment documentation using ONLY the excerpts provided below. \

Rules:
- Answer using ONLY the information in the provided excerpts. Do not use outside knowledge.
- Every factual claim in your answer MUST be followed by a citation in the form \
[filename, p.PAGE], referencing the excerpt(s) it came from.
- CRITICAL - product/document matching: Some excerpts may describe a DIFFERENT product/board \
than the one named in the question, and may coincidentally use similar terminology (e.g. \
multiple boards each have a section called "VIN Rating" or discuss power input specs). \
Before using an excerpt, check whether its SOURCE_DOCUMENT clearly corresponds to the \
product named in the question. If an excerpt's source document clearly matches the named \
product, use it confidently and normally -- do not refuse to answer just because OTHER \
excerpts in this prompt are about a different product. Only respond with "I don't know \
based on the provided documents" if NONE of the excerpts can be confidently matched to the \
product named in the question. Do not let excerpts about a different product distract you \
from a correct, clearly-matching excerpt that IS present.
- If the excerpts do not contain enough information to answer the question, say exactly: \
"I don't know based on the provided documents." Do not guess or fill gaps with outside knowledge.
- If different excerpts give conflicting information, point out the conflict explicitly \
rather than picking one silently.
- Be concise and precise. Prefer exact values/units/part numbers as written in the excerpts.
"""


def _format_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into numbered, labeled blocks for the prompt.

    SOURCE_DOCUMENT is pulled out as its own bold-style line (not just folded into
    a bracketed header) specifically to make document identity harder to miss.
    This followed a real failure: with multiple similar products in the corpus
    (e.g. two different boards each having a "VIN Rating" section), the model
    picked an excerpt by matching a keyword in the section title against the
    query, without properly checking that excerpt's source file matched the
    product actually named in the question. Making the source document its own
    visually distinct line, plus the explicit instruction in SYSTEM_PROMPT, are
    both aimed at the same failure mode from two angles: making it easier to
    notice, and explicitly telling the model to check.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = (
            f"[Excerpt {i}]\n"
            f"SOURCE_DOCUMENT: {c['source_file']}\n"
            f"PAGE: {c['page_number']}\n"
            f"SECTION: {c['section']}"
        )
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Returns {"answer": str, "used_chunks": list[dict], "raw_context": str}.
    `chunks` should already be the final reranked top-N (small, precise set) --
    NOT the full hybrid candidate pool, to keep the prompt focused and avoid
    diluting the model's attention with marginally-relevant context.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "used_chunks": [],
            "raw_context": "",
        }

    context = _format_context(chunks)
    user_prompt = f"Excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=MAX_ANSWER_TOKENS,
    )
    answer = response.choices[0].message.content.strip()

    return {"answer": answer, "used_chunks": chunks, "raw_context": context}