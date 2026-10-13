"""Synthetic claims workflow; all automated assessments are advisory."""
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END

ROOT = Path(__file__).resolve().parents[1]
POLICIES = json.loads((ROOT / "data/policies.json").read_text())
CLAIMS = [
    ("CLM-1001", "accidental", 12000, 120, 1, 1, 1),
    ("CLM-1002", "cosmetic", 1500, 30, 1, 1, 1),
    ("CLM-1003", "accidental", 8000, 50, 1, 0, 1),
]
SCHEMA = """CREATE TABLE claims (id TEXT PRIMARY KEY, incident_type TEXT, amount INTEGER,
days_since_purchase INTEGER, active_coverage INTEGER, proof_of_purchase INTEGER, incident_evidence INTEGER)"""

def database():
    db = sqlite3.connect(":memory:")
    db.execute(SCHEMA)
    db.executemany("INSERT INTO claims VALUES (?,?,?,?,?,?,?)", CLAIMS)
    return db

def claim_lookup(claim_id):
    with database() as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return dict(row) if row else None

def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def retrieve(question, k=3):
    query = tokens(question)
    scores = [(len(query & tokens(p["title"] + " " + p["text"])) / max(1, len(query)), p) for p in POLICIES]
    if os.getenv("SEMANTIC_RETRIEVAL") == "1":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode([question] + [p["title"] + " " + p["text"] for p in POLICIES], normalize_embeddings=True)
            scores = [(0.5 * lex + 0.5 * float(embeddings[0] @ embeddings[i+1]), p) for i, (lex, p) in enumerate(scores)]
        except ImportError:
            pass
    return [p for _, p in sorted(scores, key=lambda item: (-item[0], item[1]["id"]))[:k]]

def rules(claim):
    if claim is None:
        return "not_found", ["Claim ID not found"]
    if not claim["proof_of_purchase"] or not claim["incident_evidence"]:
        return "needs_human_review", ["Required evidence missing"]
    if not claim["active_coverage"] or claim["days_since_purchase"] > 365:
        return "excluded_by_sample_rules", ["Coverage inactive or outside sample time limit"]
    if claim["incident_type"] in ("intentional", "cosmetic"):
        return "excluded_by_sample_rules", ["Sample exclusion applies"]
    if claim["amount"] > 50000:
        return "needs_human_review", ["Amount exceeds sample coverage limit"]
    return "potentially_eligible", ["Sample checks passed; human decision required"]

class State(TypedDict, total=False):
    claim_id: str
    question: str
    prompt_variant: str
    evidence: list
    claim: dict | None
    status: str
    reasons: list
    answer: str
    citations: list
    human_review: bool
    tool_success: bool

PROMPTS = {
    "strict": "Use only provided evidence. Cite policy IDs. Never claim final approval.",
    "concise": "Answer briefly using only evidence and policy citations; never claim final approval."
}

def retrieval_node(s):
    # Include governing policy even when lexical search misses it.
    evidence = retrieve(s["question"], 3)
    if not any(p["id"] == "POL-04" for p in evidence):
        evidence[-1] = next(p for p in POLICIES if p["id"] == "POL-04")
    return {"evidence": evidence}

def tools_node(s):
    claim = claim_lookup(s["claim_id"])
    status, reasons = rules(claim)
    return {"claim": claim, "status": status, "reasons": reasons, "tool_success": claim is not None}

def compose_node(s):
    ids = [p["id"] for p in s["evidence"]]
    # Fixed, inspectable fallback. Optional LLM may summarize but cannot override status.
    answer = f"Advisory sample-rule status: {s['status']}. {'; '.join(s['reasons'])}. Final resolution requires a human reviewer. Sources: " + ", ".join(f"[{i}]" for i in ids) + "."
    if os.getenv("OPENAI_API_KEY"):
        try:
            import httpx
            context = "\n".join(f"{p['id']}: {p['text']}" for p in s["evidence"])
            prompt = PROMPTS.get(s.get("prompt_variant", "strict"), PROMPTS["strict"])
            response = httpx.post(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "temperature": 0,
                      "messages": [{"role":"system","content":prompt + " Do not override the tool status or invent citations."},
                                   {"role":"user","content":f"Question: {s['question']}\nTool status: {s['status']}\nReasons: {s['reasons']}\nEvidence:\n{context}"}]},
                timeout=20)
            response.raise_for_status()
            candidate = response.json()["choices"][0]["message"]["content"]
            used = re.findall(r"\[(POL-\d+)\]", candidate)
            if used and all(i in ids for i in used) and s["status"] in candidate:
                answer = candidate
        except (ValueError, KeyError, httpx.HTTPError):
            pass
    return {"answer": answer}

def verify_node(s):
    cited = re.findall(r"\[(POL-\d+)\]", s["answer"])
    valid = {p["id"] for p in s["evidence"]}
    if not cited or any(c not in valid for c in cited):
        return {"answer": "Insufficient verifiable citations. Human review required.", "citations": [], "human_review": True}
    return {"citations": list(dict.fromkeys(cited)), "human_review": s["status"] != "potentially_eligible"}

builder = StateGraph(State)
for name, fn in [("retrieve", retrieval_node), ("tools", tools_node), ("compose", compose_node), ("verify", verify_node)]:
    builder.add_node(name, fn)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "tools")
builder.add_edge("tools", "compose")
builder.add_edge("compose", "verify")
builder.add_edge("verify", END)
graph = builder.compile()

def ask(claim_id, question, prompt_variant="strict"):
    started = time.perf_counter()
    state = graph.invoke({"claim_id": claim_id, "question": question, "prompt_variant": prompt_variant})
    return {key: state[key] for key in ("claim_id", "question", "status", "reasons", "answer", "citations", "human_review", "tool_success")} | {"retrieved_ids": [p["id"] for p in state["evidence"]], "latency_ms": round((time.perf_counter()-started)*1000, 2)}
