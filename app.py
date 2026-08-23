import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_code import AppConfig, NLI_MODEL, NLIVerifier, build_retrieval_stack, QAEngine, format_answer  # noqa: E402

app = FastAPI(title="Pulser — Cardiac Health Guide")

_cfg = AppConfig.from_yaml(Path("config/config.yaml"))
_retriever = None
_engine = None


def _get_engine():
    global _retriever, _engine
    if _engine is None:
        from groq import Groq
        _retriever = build_retrieval_stack(_cfg)
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        nli_verifier = NLIVerifier(NLI_MODEL, device=_cfg.embeddings.device)
        _engine = QAEngine(_retriever, groq_client, nli_verifier=nli_verifier)
    return _engine


class QuestionPayload(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask_endpoint(payload: QuestionPayload):
    engine = _get_engine()
    result = engine.ask(payload.question)

    return {
        "refused": result.gate_decision == "refused",
        "overall_level": result.overall_level.value if result.overall_level else "Unsupported",
        "overall_score": round(result.overall_score, 4) if result.overall_score else 0.0,
        "claims": [
            {
                "text": c.text,
                "evidence": c.evidence or [],
                "citations": c.citations or [],
            }
            for c in (result.claims or [])
            if not c.dropped
        ],
        "follow_up_questions": result.follow_up_questions or [],
        "suggested_alternative": result.suggested_alternative,
        "disclaimer": "This guide provides evidence-based research info and is not a substitute for professional medical advice.",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
