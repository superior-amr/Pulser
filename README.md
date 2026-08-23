# Cardiovascular Medical RAG

A professional, modular Retrieval-Augmented Generation pipeline over a cardiovascular medical corpus, built from a Colab notebook into a production-grade Python project. It ingests six cardiovascular PDFs (lectures, NHS medicines guidance, a clinical review, and reference textbooks), builds a clean semantically chunked index, retrieves with a **hybrid dense + BM25 + RRF fusion** stack, reranks with a cross-encoder, and answers with an LLM layer that enforces evidence-gated, refusal-aware, claim-verified responses.

After 14 development rounds, the system achieves **94% rank-1 accuracy on a 50-question adversarial test** (up from ~40% on the original 20-question baseline), with **100% out-of-scope refusal** on the QA safety test.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Pipeline Stages](#pipeline-stages)
7. [QA Answering Layer](#qa-answering-layer)
8. [Development History and Results](#development-history-and-results)
9. [Requirements](#requirements)
10. [License](#license)

---

## Project Structure

```
rag_project/
├── main.py                      # CLI entry point
├── run_qa_tests.py              # QA safety regression tests
├── src/
│   ├── ingestion.py             # Docling PDF parsing → DoclingDocument
│   ├── cleaning.py              # Garbage/copyright filtering, front-matter removal
│   ├── chunking.py              # Docling-aware structural units → semantic chunks
│   ├── nlp_utils.py             # spaCy sentence utilities
│   ├── models.py                # ProcessedChunk data model (serialization)
│   ├── enrich.py                # Acronym expansion, metadata enrichment
│   ├── embeddings.py            # Dual-embedder index build (cached .npz)
│   ├── retrieval.py             # Hybrid retriever: dense + BM25 + RRF fusion + reranker
│   ├── evaluation.py            # Load questions + retriever config sweep
│   ├── pipeline.py              # 6-stage orchestrator with checkpointing
│   ├── config.py                # YAML-backed central configuration
│   └── qa/
│       ├── engine.py            # QAEngine: gated multi-query generation + claim verification
│       ├── multi_query.py       # Query expansion with deterministic fallback
│       ├── confidence.py        # Confidence ladder (0.77 → 0.96)
│       ├── prompts.py           # Claims-format system prompt + refusal message
│       ├── logger.py            # JSON trace logging
│       └── ask_cli.py           # Interactive / one-shot QA CLI
├── config/
│   ├── config.yaml              # All thresholds, models, and sweep parameters
│   └── acronyms.yaml            # Medical acronym expansion dictionary
├── data/                        # Source PDFs + questions.json
├── output/                      # processed_semantic_chunks.json, eval results
├── cache/                       # embeddings.npz (index cache)
└── qa_logs/                     # Per-question trace JSONs
```

## Architecture Overview

The system follows a strict two-tier design. The **indexing tier** (`src/ingestion.py` → `src/cleaning.py` → `src/chunking.py` → `src/enrich.py` → `src/embeddings.py`) converts PDFs into a clean chunk index, and the **retrieval and answering tier** (`src/retrieval.py`, `src/qa/`) consumes it. All configuration lives in `config/config.yaml` and is read through a single `AppConfig` object; nothing is hardcoded in source.

```
PDFs → Docling (layout parsing + tables + OCR fallback)
  → garbage/copyright filter
  → structural units (item.label sections, per-line page provenance)
  → spaCy-aware semantic chunking (percentile 60, overlap control)
  → acronym expansion + metadata enrichment (31 fields/chunk)
  → BGE-base embeddings → cached index (.npz)

Question → multi-query expansion (5 variants)
  → per-variant hybrid retrieval (dense + BM25 → RRF fusion)
  → merge + rerank (bge-reranker-base)
  → rejection gate (0.77) → LLM generation (Groq, claims format)
  → claim verification (numeric grounding checks)
  → formatted answer: Recommendation / Evidence / Citation / Uncertainty
```

## Installation

Requires Python 3.11+ and Anaconda (or any pip environment).

```bash
conda create -n rag python=3.11 -y && conda activate rag

pip install docling pypdf langchain langchain-experimental langchain-core
pip install sentence-transformers spacy numpy pyyaml groq scikit-learn

python -m spacy download en_core_web_sm
```

On Windows, enable **Developer Mode** (Settings → Update & Security → For developers) before the first run: Docling's model cache uses symlinks, which are blocked for ordinary users otherwise. A one-time download of 200–600 MB of models occurs on first run. Set the LLM key for the QA layer:

```powershell
$env:GROQ_API_KEY = 'gsk_xxxxx...'
```

A free API key is available at [groq.com](https://groq.com). The indexing tier runs fully offline without it.

## Configuration

Every threshold and model choice lives in `config/config.yaml`. The most important blocks:

| Block | Key settings | Current values |
|---|---|---|
| `parsing` | `use_ocr`, `table_mode`, `do_table_structure` | `true`, `accurate`, `true` |
| `preprocessing` | `tiny_chunk_char_limit`, `min_chunk_chars`, `ideal_max_chars` | 250, 250, 1800 |
| `chunking` | `semantic_percentile` | 60.0 (percentile method) |
| `embeddings` | `index_embedder`, `reranker_model` | `BAAI/bge-base-en-v1.5`, `BAAI/bge-reranker-base` |
| `retrieval` | `rerank_k`, `mmr_diversity` | 10, 0.3 |

Retrieval defaults are `similarity` with `k=5`, pinned after 13 rounds of config sweeps in which this combination won consistently. The question set lives in `data/questions.json`.

## Usage

```bash
# Full pipeline: ingest → chunk → enrich → embed → eval (sweep if no saved config)
python main.py run

# Individual stages
python main.py process   # rebuild the chunk index from data/*.pdf
python main.py embed     # embed chunks; run retriever config sweep
python main.py eval      # run the saved retriever on data/questions.json

# Dense retrieval only (interactive, over the index)
python main.py serve

# LLM answering with evidence gating (GROQ_API_KEY required)
python src/qa/ask_cli.py                      # interactive loop
python src/qa/ask_cli.py "What determines cardiac output?"   # one-shot

# QA safety regression tests
python run_qa_tests.py
```

Chunk index rebuilds take roughly 3–7 minutes (one-time cost per corpus change); subsequent retrievals are sub-second, with reranking adding a few seconds per batch.

## Pipeline Stages

**Ingestion.** Docling's `DocumentConverter` parses each PDF with layout analysis, exports tables as Markdown, extracts image captions, and falls back to RapidOCR for scanned pages. Each extracted element carries per-line page provenance.

**Cleaning.** Removes front matter, header/footer repetition, table-of-contents entries, and a dedicated garbage/copyright filter that drops lines matching patterns such as `unauthorized use prohibited` and document tracking codes (e.g., `WF618229`).

**Chunking.** `build_structural_units_from_docling` walks the document tree using `item.label` for section headers and `prov[0].page_no` for page numbers; structural units are grouped, then split by LangChain's semantic chunker at the 60th percentile of sentence embedding distance, with overlap and size-control sweeps applied during development. A final safety cleanup strips standalone section-number artifacts.

**Enrichment.** Chunks receive 31 metadata fields (source file, pages, section title, parser type, acronym expansion, visual references) and a normalized section title scheme (e.g., `❖ N.B:` markers become "High-Yield Notes").

**Embeddings.** A dual-embedder design uses cheap MiniLM solely for semantic boundary detection and `BAAI/bge-base-en-v1.5` for the retrieval index; the matrix is cached as `embeddings.npz` and reloaded on subsequent runs.

**Retrieval.** Three-stage hybrid retrieval: dense cosine search over the normalized embedding matrix, BM25 sparse scoring over query tokens, and RRF rank fusion (`1/(60+rank)`), followed by `BAAI/bge-reranker-base` cross-encoder reranking. Dense search and BM25 each fetch up to `k`, fused, then reranked with `rerank_k` candidates.

## QA Answering Layer

The `src/qa/` package sits above the retrieval stack and calls `build_retrieval_stack` unchanged. For each question, `QAEngine.ask` executes a ten-step flow: multi-query expansion into five search variants, per-variant hybrid retrieval, result merging by best score, a **rejection gate at 0.77** (questions below the gate are refused before the LLM speaks), Groq generation constrained by a claims-format system prompt, numeric grounding verification of every claim, and a formatted final answer with Recommendation, Evidence, Citation, and Uncertainty sections.

Confidence is reported on a legalistic ladder: `Unsupported (<0.77) → Weak (0.77) → Partial (0.83) → Well (0.89) → Strong (0.96)`. Every call writes a full JSON trace to `qa_logs/` for auditability. The layer deliberately fetches `k=20` at retrieval time to capture table and guideline chunks the reranker favors, while evaluation uses the tuned `k=5`.

## Development History and Results

The project evolved through 14 documented rounds from a single Colab notebook to a modular CLI application:

| Rounds | Era | Key work | Result |
|---|---|---|---|
| 1–10 | unstructured.io | Notebook → professional architecture; spaCy sentence splitting; 15% overlap; header detection; semantic chunking fixes (mega-chunks, mid-sentence cuts, metadata corruption); reranker swap MiniLM → bge-reranker; garbage filter; pinned k=5 | Index 554 clean chunks; 8/20 questions rank-1 correct |
| 11–12 | Docling migration | Parser swap for layout-aware headers, per-line page numbers, table Markdown, picture captions; Windows fixes (symlink privileges, RapidOCR model, PictureItem errors); marker normalization | Index 689 chunks, all 5 PDFs, 0 garbage; titles normalized |
| 13 | Hybrid retrieval | BM25 token weights + RRF rank fusion added to `retrieve`; garbage filter in chunking | 36/50 (72%) rank-1 on the 50-question adversarial set |
| 14 | Expansion + QA layer | Sixth corpus file added; QA engine with rejection gating and claim verification | **47/50 (94%) rank-1; 49/50 in top-3; 100% out-of-scope refusal (5/5)** |

Notable individual gains: MRA therapy moved from a copyright-boilerplate chunk to the correct drug-table chunk (+0.996); Class Ic antiarrhythmics, CKM syndrome, WPW, Brugada, and ezetimibe all moved from persistent failure to rank-1 correct after the corpus expansion and hybrid retrieval.

## Requirements

| Component | Version / notes |
|---|---|
| Python | 3.11+ |
| Core libraries | `docling`, `pypdf`, `langchain`, `langchain-experimental`, `sentence-transformers`, `spacy`, `numpy`, `pyyaml`, `groq`, `scikit-learn` |
| spaCy model | `en_core_web_sm` |
| Embeddings | `BAAI/bge-base-en-v1.5` (index), `all-MiniLM-L6-v2` (chunking boundaries) |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | Groq (requires `GROQ_API_KEY`); indexing tier fully offline |
| OS note | Windows: enable Developer Mode for Docling model symlinks |

## License

Educational project built on open-source components: [Docling](https://github.com/docling-project/docling) (MIT), [BAAI embedding models](https://huggingface.co/BAAI) (MIT), [Groq](https://groq.com) (API terms). Corpus PDFs remain the property of their respective authors and are used here for personal educational purposes only.

---

*Built and documented across 14 iterative development rounds. Full per-round experiment log is preserved in the project's iteration tracker spreadsheet.*  




⚖️ License & Legal Notice
This project is licensed under the PolyForm Noncommercial License 1.0.0.

The source code is freely available for personal, educational, and academic research purposes only.

Strictly Prohibited: Any commercial use, monetization, or integration into for-profit products is strictly forbidden without explicit prior written permission from the original author.

Attribution & Modifications: Any permitted use of this code must include clear attribution to the original author. If you modify the code, you must explicitly state that changes were made to the original work.

For commercial licensing inquiries, please contact me directly.


⚖️ حقوق الملكية والترخيص
هذا المشروع يخضع بالكامل لترخيص PolyForm Noncommercial 1.0.0.

الكود المصدري متاح مجاناً للاستخدام الشخصي، التعليمي، والبحثي فقط.

الاستخدام التجاري: يُمنع منعاً باتاً استخدام هذا الكود لأي أغراض تجارية، أو التربح منه، أو دمجه ضمن أي منتج تجاري دون الحصول على إذن كتابي مسبق مني كصاحب المشروع.

النسب والتعديل: أي استخدام مصرح به لهذا المشروع يجب أن يتضمن إشارة واضحة للمؤلف الأصلي. وفي حال قمت بإجراء أي تعديلات على الكود، يجب عليك توضيح ذلك صراحةً.

للحصول على ترخيص تجاري، يُرجى التواصل معي مباشرة.


### I am open to work and develope this project together and helping to improve human's health.

---

## 📬 Contact & Commercial Licensing

If you are a company or an organization interested in using this project for commercial purposes, or if you need a custom commercial license, please feel free to reach out to me directly:

*   **Email:** [amrshalapy101@gmail.com](mailto:amrshalapy101@gmail.com)
*   **LinkedIn:** [linkedin.com/in/Amr Gamal Shalapy](https://www.linkedin.com/in/amr-gamal-shalapy-0b839a27a?utm_source=share_via&utm_content=profile&utm_medium=member_android)
*   **GitHub:** [github.com/superior-amr](https://github.com/superior-amr)

Let's discuss how we can work together!