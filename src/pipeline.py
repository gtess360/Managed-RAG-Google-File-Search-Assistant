"""End-to-end RAG query pipeline for the Personal AI Knowledge Assistant.

Powered by the Google Gemini API File Search tool, which performs chunking, embedding,
retrieval and grounded generation server-side:
1. Validates the user question string.
2. Sends it to Gemini with the managed File Search store attached.
3. Maps grounding metadata to citations and retrieved evidence (never trusting
   model-invented source paths).
4. Returns answered / partial / conflict / abstain / error with explicit abstention
   when evidence is insufficient.
5. Returns a clean structured dictionary consumable by the CLI, FastAPI and Streamlit.
"""

from __future__ import annotations

import time
from typing import Any, Optional
from pathlib import Path

from src.file_search import (
    ask,
    INSUFFICIENT_EVIDENCE_PHRASE,
    CITATION_VALIDITY_DISCLAIMER,
)
from src.config import (
    DEFAULT_TOP_K,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    get_gemini_timeout_seconds,
)

MAX_QUESTION_CHARS = 500


def validate_query(question: str) -> str:
    """Return the stripped question or raise ValueError if empty or too long."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty or whitespace only.")
    cleaned = question.strip()
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question exceeds the {MAX_QUESTION_CHARS}-character limit.")
    return cleaned


def query_pipeline(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    storage_dir: Optional[Path] = None,  # kept for backward compatibility; unused
    client: Optional[Any] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: int = 2,  # kept for backward compatibility; unused
) -> dict:
    """Execute the RAG pipeline via Gemini File Search.

    Args:
        question: User query string.
        top_k: Number of evidence chunks File Search should retrieve.
        storage_dir: Deprecated (local Chroma storage was removed); ignored.
        client: Optional google-genai client (e.g. for offline mocking).
        model: Optional Gemini model identifier override.
        timeout: Request timeout in seconds (defaults to GEMINI_TIMEOUT_SECONDS or 120s).
        max_retries: Deprecated; ignored.

    Returns:
        Structured result dictionary:
            question: str
            answer: str
            status: "answered" | "abstain" | "partial" | "conflict" | "error"
            missing_information: Optional[str]
            citations: list[dict] (chunk_id, source, page from grounding metadata)
            invalid_citations: list[str]
            citation_disclaimer: str
            retrieved_evidence: list[dict] (distance is always None for File Search)
            timings: dict with retrieval_ms, generation_ms, total_ms
            model: str
            error: Optional[str]
    """
    t0 = time.perf_counter()
    target_model = (model or GEMINI_MODEL).strip()

    try:
        clean_question = validate_query(question)
    except ValueError as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "question": question,
            "answer": "",
            "status": "error",
            "missing_information": None,
            "citations": [],
            "invalid_citations": [],
            "citation_disclaimer": "",
            "retrieved_evidence": [],
            "timings": {"retrieval_ms": 0.0, "generation_ms": 0.0, "total_ms": round(total_ms, 2)},
            "model": target_model,
            "error": str(e),
            "error_details": {
                "exception_type": "ValueError",
                "code": 400,
                "model": target_model,
                "sanitized_reason": str(e),
                "summary": "Validation error: invalid question input.",
            },
        }

    target_timeout = timeout if timeout is not None else get_gemini_timeout_seconds()

    return ask(
        question=clean_question,
        top_k=top_k,
        model=target_model,
        client=client,
        timeout=target_timeout,
    )


def main() -> None:
    """CLI runner for testing the RAG pipeline on a question."""
    import sys

    default_question = "What embedding model and vector database were used in Project Nexus?"
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else default_question

    print("=" * 70)
    print("PERSONAL AI KNOWLEDGE ASSISTANT — PIPELINE QUERY")
    print("=" * 70)
    print(f"Question: {question}\n")

    result = query_pipeline(question)

    print(f"Status: {result['status']}")
    print(f"Model:  {result['model']}")
    print(
        f"Timings: Retrieval={result['timings']['retrieval_ms']}ms | "
        f"Generation={result['timings']['generation_ms']}ms | "
        f"Total={result['timings']['total_ms']}ms\n"
    )
    print("ANSWER:")
    print(result["answer"])
    print()

    if result.get("missing_information"):
        print(f"Missing Information: {result['missing_information']}\n")

    if result["citations"]:
        print("CITATIONS (Resolved from Storage Metadata):")
        for cit in result["citations"]:
            page_str = f", Page {cit['page']}" if cit.get("page") else ""
            print(f"  - [{cit['chunk_id']}] -> Source: {cit['source']}{page_str}")
        print()

    if result["invalid_citations"]:
        print(f"INVALID / UNRESOLVED CITATIONS: {result['invalid_citations']}\n")

    print("RETRIEVED EVIDENCE:")
    for i, ev in enumerate(result["retrieved_evidence"], 1):
        page_str = f", Page {ev['page']}" if ev.get("page") else ""
        dist_str = f"{ev['distance']:.4f}" if ev.get("distance") is not None else "N/A"
        print(f"  {i}. [{ev['chunk_id']}] (Distance: {dist_str}, Source: {ev['source']}{page_str})")
        print(f"     Excerpt: {ev['text'][:120]}...\n")

    if result.get("error"):
        print(f"ERROR: {result['error']}\n")

    print(f"NOTE: {result['citation_disclaimer']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
