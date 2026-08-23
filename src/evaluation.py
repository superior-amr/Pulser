"""Final retrieval evaluation export (notebook Block 12).

Runs the selected retriever configuration over the evaluation question set
and writes a self-describing JSON document (evidence + chunk metadata) that
can be consumed by notebooks, dashboards, or CI checks.

Question source: external JSON file (default ``data/questions.json``),
overridable via ``load_questions(path)`` or the ``QUESTIONS_PATH`` default.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .retrieval import Retriever

# Path to the external question file (relative to the project root).
QUESTIONS_PATH: str = "data/questions.json"


def load_questions(path: str | Path = QUESTIONS_PATH) -> List[Dict[str, str]]:
    """Load the question set from an external JSON file.

    The file must contain a JSON array of objects, each with at least:
        {"question": "...", "keywords": ["...", "..."]}
    (an optional "note" field per question is allowed but ignored).
    """
    qpath = Path(path)
    if not qpath.is_absolute():
        qpath = Path.cwd() / qpath
    if not qpath.exists():
        raise FileNotFoundError(
            f"Question file not found: {qpath}. Place the 50-question set at "
            f"data/questions.json (see questions_50.json)."
        )
    with qpath.open(encoding="utf-8") as fh:
        questions = json.load(fh)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Question file must contain a non-empty JSON array: {qpath}")
    for i, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or "question" not in item:
            raise ValueError(
                f"Question file entry #{i} is missing a 'question' field: {qpath}"
            )
        item.setdefault("keywords", [])
    print(f"    Loaded {len(questions)} questions from {qpath}")
    return questions


def run_evaluation(
    retriever: Retriever,
    questions: List[Dict[str, str]],
    embedding_model_name: str,
    reranker_model_name: str,
    output_path: Path,
) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "retriever_config": {
            "search_type": retriever.cfg.search_types[0],
            "k": retriever.cfg.k_values[0],
            "rerank_k": retriever.cfg.rerank_k,
        },
        "questions": [],
    }

    for item in questions:
        results = retriever.retrieve(item["question"], retriever.cfg.k_values[0],
                                     retriever.cfg.rerank_k)
        report["questions"].append({
            "question": item["question"],
            "keywords": item.get("keywords", []),
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "dense_score": round(r.dense_score, 4),
                    "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                    "source_file": r.chunk.metadata.source_file,
                    "section_title": r.chunk.metadata.section_title,
                    "page_numbers": r.chunk.metadata.page_numbers,
                    "text": r.chunk.original_text,
                    "expanded_text": r.chunk.expanded_text,
                }
                for r in results
            ],
        })

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"    Evaluation report saved: {output_path}")
    return report
