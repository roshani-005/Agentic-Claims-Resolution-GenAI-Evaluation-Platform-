# ClaimAssist: Agentic Claims Resolution & GenAI Evaluation Platform

ClaimAssist is an educational, synthetic-data platform for document-assisted claim review. It combines OCR, policy retrieval, deterministic SQL-backed checks, a LangGraph workflow, citation verification, a human-review gate and repeatable evaluation.

> **Safety boundary:** all claims and policies are invented. Outputs are advisory only. This is not an insurance product and must never approve, deny or price a real claim.

## Implemented capabilities

- Optional PaddleOCR image extraction, conservative field candidates, PP-Structure table detection and Character Error Rate helper.
- Lexical retrieval with optional Sentence-Transformers semantic ranking and reciprocal-rank fusion. No policy is silently injected.
- Parameterized SQLite claim lookup, deterministic sample rules and persistent synthetic records.
- LangGraph retrieval → SQL/rules → response composition → citation verification. Every result requires human review.
- Timestamped audit records containing status, reasons, retrieved IDs, citations and latency.
- Synthetic evaluation: status accuracy, retrieval recall@3, citation precision, unsupported citation rate, human-review rate, tool success, mean latency and p95 latency.
- FastAPI health, claim lookup, question answering, OCR, audit and evaluation endpoints.

## Architecture

```text
Image upload → PaddleOCR → extracted text / candidate fields / detected tables
Question + Claim ID → LangGraph
  → Hybrid policy retrieval
  → Parameterized SQLite claim lookup
  → Deterministic sample eligibility checks
  → Evidence-grounded response
  → Citation verification
  → Mandatory human-review flag
  → SQLite audit log → Evaluation / FastAPI
```

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

```bash
pytest -q
python -m app.evaluate
```

## Optional OCR

PaddlePaddle wheels vary by operating system and CUDA version. Install separately:

```bash
pip install -r requirements-ocr.txt
```

Send a PNG/JPG/BMP/TIFF image to `POST /documents/extract` as multipart form data. The response includes OCR text, confidence values, candidate fields and detected table HTML. OCR output is untrusted and requires human verification. PDF rasterization is not included.

## Optional semantic retrieval and LLM

Install `sentence-transformers` and set `SEMANTIC_RETRIEVAL=1` for semantic ranking. The first request downloads the configured embedding model if it is not cached. Missing optional dependencies fall back to lexical retrieval.

Copy `.env.example` to `.env`. Set credentials outside source control. Set `ENABLE_LLM=1` to enable the OpenAI-compatible endpoint. Deterministic rules own the claim status; LLM text never overrides tool findings. Offline deterministic mode is the default.

## API examples

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"claim_id":"CLM-1001","question":"Is accidental damage potentially covered?"}'
curl http://127.0.0.1:8000/audit?limit=20
curl http://127.0.0.1:8000/evaluate
```

## Evaluation and limitations

The fixed cases in `data/eval_cases.json` use manually assigned relevant policy IDs. Citation precision checks ID membership in retrieved evidence; it does not prove semantic entailment. The benchmark is tiny and synthetic, not statistically meaningful production validation. p95 latency is environment-dependent. Cost is zero only in offline deterministic mode; provider cost requires token telemetry and current pricing.

The sample policy is invented and is not legal or insurance guidance. OCR table cells are not reconciled against a claim schema. Production use would require authentication, authorization, rate limiting, encryption, retention controls, monitoring and security review.

The local SQLite database is created at `data/claimassist.db` and ignored by Git. Set `CLAIMASSIST_DB` to store it elsewhere. Never upload real medical records or customer data.
