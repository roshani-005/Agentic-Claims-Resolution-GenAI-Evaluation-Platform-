# ClaimAssist — Agentic Claims Resolution & GenAI Evaluation

An educational, synthetic-data claims assistant: retrieval, citations, deterministic eligibility tools, human review, optional LLM synthesis, and repeatable evaluation. **Never use for real coverage decisions.**

## Features
- Hybrid lexical + optional Sentence-Transformers semantic retrieval with reranking
- LangGraph workflow (retrieve → policy/SQL tools → draft → verify → human review gate)
- SQLite synthetic claims, parameterized queries, explicit eligibility checks
- FastAPI endpoints: `/health`, `/claims/{claim_id}`, `/ask`, `/evaluate`
- Optional OpenAI-compatible LLM via environment variables; deterministic offline fallback
- Fixed evaluation cases and metrics: retrieval recall, citation precision, abstention, tool success, latency, token/cost estimates
- Prompt variants, structured evidence, audit-friendly outputs, no raw personal data

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -q
python -m app.evaluate
```
Optional semantic retrieval: `pip install sentence-transformers` and set `SEMANTIC_RETRIEVAL=1`. Model download requires internet. Optional LLM: set `OPENAI_API_KEY`, `OPENAI_MODEL`; no key means offline evidence-first responses. Do not put credentials in source control.

## Example
```bash
curl -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"claim_id":"CLM-1001","question":"Is this claim eligible under the sample policy?"}'
```
All policies and claims are invented. A rules result is a demonstration, not a real adjudication. Human review is required for missing/ambiguous evidence and any consequential decision.

## Architecture
`API → LangGraph: retrieve → lookup + rules → compose → citation verification → review gate → response`.
The deterministic fallback is intentionally transparent. LLM output is optional, constrained to cited evidence and never overrides tool findings.

## Evaluation
`python -m app.evaluate` writes JSON results to stdout. Fixed cases in `data/eval_cases.json` compare prompt variants without changing the underlying questions. Reported API cost is an estimate when token usage is available, not a billing record.

## JD mapping
Python/FastAPI, data preprocessing, NLP/embeddings, prompt evaluation, debugging, API integration, monitoring, documentation, privacy and responsible AI. Computer Vision/Speech are **not claimed as implemented**; future extension: OCR intake and speech transcription.
