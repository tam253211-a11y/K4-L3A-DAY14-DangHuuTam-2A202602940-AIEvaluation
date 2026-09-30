"""Ask the OrbitTech RAG assistant a question and inspect its retrieval trace.

Usage:
    python ask.py "Can I return opened ear tips?"
    python ask.py            # interactive mode, empty line to quit

When the question matches a golden-dataset question, the five evaluation
metrics from template.py are included as well. ``chat_app.py`` serves the same
result in a browser chat.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAIError

from domain_assistant import DomainAssistant
from template import RAGASEvaluator

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "technology_store"
GOLDEN_PATH = ROOT / "golden_dataset.json"
PREVIEW_CHARS = 110


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_golden() -> dict[str, dict[str, Any]]:
    """Map each normalized golden question to its dataset record."""
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return {normalize_question(pair["question"]): pair for pair in dataset["qa_pairs"]}


def answer_question(
    assistant: DomainAssistant,
    golden: dict[str, dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    """Run the assistant on one question and return answer, trace and scores.

    ``scores`` is None unless the question is a golden-dataset question.
    Raises OpenAIError, RuntimeError or ValueError when generation fails.
    """
    response = assistant.answer_with_trace(question)
    chunks = [
        {
            "rank": rank,
            "source_doc": chunk.source_doc,
            "chunk_id": chunk.chunk_id,
            "score": round(chunk.score, 2),
            "text": chunk.text,
        }
        for rank, chunk in enumerate(response.retrieved_chunks, start=1)
    ]

    scores: dict[str, Any] | None = None
    pair = golden.get(normalize_question(question))
    if pair is not None:
        result = RAGASEvaluator().run_full_eval(
            answer=response.actual_answer,
            question=question,
            context="\n\n".join(context["text"] for context in pair["contexts"]),
            expected=pair["expected_answer"],
            contexts=[chunk["text"] for chunk in chunks],
        )
        scores = {
            "id": pair["id"],
            "difficulty": pair["difficulty"],
            "expected_answer": pair["expected_answer"],
            "context_recall": result.context_recall,
            "context_precision": result.context_precision,
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "completeness": result.completeness,
            "overall": result.overall_score(),
            "passed": result.passed,
            "failure_type": result.failure_type,
        }

    return {"answer": response.actual_answer, "chunks": chunks, "scores": scores}


def _print_result(result: dict[str, Any]) -> None:
    print("\nANSWER")
    print(result["answer"])

    print(f"\nRETRIEVED CHUNKS ({len(result['chunks'])})")
    for chunk in result["chunks"]:
        preview = re.sub(r"\s+", " ", chunk["text"])[:PREVIEW_CHARS]
        print(
            f"  {chunk['rank']}. [{chunk['source_doc']} | {chunk['chunk_id']} "
            f"| BM25 {chunk['score']:.2f}]"
        )
        print(f"     {preview}...")

    scores = result["scores"]
    if scores is None:
        print("\n(No golden answer for this question, so no scores.)")
        return
    print(f"\nSCORES (golden case {scores['id']}, {scores['difficulty']})")
    print(f"  Expected: {scores['expected_answer']}")
    print(
        f"  Context Recall {scores['context_recall']:.3f} | "
        f"Context Precision {scores['context_precision']:.3f}"
    )
    print(
        f"  Faithfulness {scores['faithfulness']:.3f} | "
        f"Relevance {scores['relevance']:.3f} | "
        f"Completeness {scores['completeness']:.3f} | "
        f"Overall {scores['overall']:.3f}"
    )
    print(f"  Passed: {scores['passed']} | Failure type: {scores['failure_type'] or '-'}")


def ask(
    assistant: DomainAssistant,
    golden: dict[str, dict[str, Any]],
    question: str,
) -> None:
    try:
        result = answer_question(assistant, golden, question)
    except (OpenAIError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return
    _print_result(result)


def main() -> int:
    try:
        assistant = DomainAssistant.from_corpus(CORPUS_DIR)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    golden = load_golden()

    if len(sys.argv) > 1:
        ask(assistant, golden, " ".join(sys.argv[1:]))
        return 0

    print("OrbitTech RAG assistant. Type a question, or press Enter to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(assistant, golden, question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
