"""Run the 20-question mixed QA test.

Usage (from the rag_project root, with GROQ_API_KEY set):
    python run_test_20.py data/questions_test_20.json

Expected behavior encoded in this script:
- Questions 1-15 (in-scope)  -> should be answered (gate_decision == "generated"),
                                  overall_score >= REFUSE_THRESHOLD
- Questions 16-20 (out-of-scope) -> should be refused (gate_decision == "refused")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from full_code import (
    AppConfig,
    NLI_MODEL,
    NLIVerifier,
    QAEngine,
    TraceLogger,
    build_retrieval_stack,
    format_answer,
    REFUSE_THRESHOLD,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_test_20.py <questions_json_path>")
        sys.exit(1)

    questions_path = Path(sys.argv[1])
    if not questions_path.exists():
        print(f"File not found: {questions_path}")
        sys.exit(1)

    with questions_path.open(encoding="utf-8") as fh:
        questions = json.load(fh)

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Set it first.")
        sys.exit(1)

    from groq import Groq

    cfg = AppConfig.from_yaml(Path("config/config.yaml"))
    retriever = build_retrieval_stack(cfg)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    nli_verifier = NLIVerifier(NLI_MODEL, device=cfg.embeddings.device)
    engine = QAEngine(retriever, groq_client, nli_verifier=nli_verifier)

    LOG_DIR = Path(cfg.paths.output_dir) / "qa_logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = len(questions)

    print(f"\nRunning {total}-question QA test...\n")

    for i, q in enumerate(questions, start=1):
        question = q["question"]
        case_id = q.get("id", f"Q{i:03d}")
        expected_refuse = q.get("expect_refuse", False)

        trace_path = LOG_DIR / f"test_{case_id}_trace.json"
        logger = TraceLogger(path=str(trace_path))
        engine.logger = logger

        result = engine.ask(question, show_details=False)
        logger.set("result_summary", {
            "gate": result.gate_decision,
            "overall_score": result.overall_score,
            "overall_level": result.overall_level.value if result.overall_level else None,
            "n_claims": len(result.claims),
            "validation": result.validation,
        })
        logger.save()

        refused = result.is_refused
        if expected_refuse:
            ok = refused
            verdict = "REFUSE-OK" if ok else "SHOULD-HAVE-REFUSED"
        else:
            ok = (not refused) and (
                result.overall_score is not None
                and result.overall_score >= REFUSE_THRESHOLD
            )
            verdict = "ANSWER-OK" if ok else "WEAK-OR-REFUSED"

        if ok:
            correct += 1
        score_str = (
            f"{result.overall_score:.3f} ({result.overall_level.value})"
            if result.overall_score is not None
            else "n/a (refused)"
        )

        print(f"[{i:2d}/{total}] {case_id} | {verdict:20s} | score={score_str}")
        print(f"         Q: {question[:80]}...")
        if result.follow_up_questions:
            print(f"         Follow-ups: {len(result.follow_up_questions)}")
        if result.suggested_alternative:
            print(f"         Alternative: {result.suggested_alternative[:60]}...")
        print()

    accuracy = correct / total if total > 0 else 0.0
    print(f"{'=' * 80}")
    print(f"RESULT: {correct}/{total} correct ({accuracy:.1%})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
