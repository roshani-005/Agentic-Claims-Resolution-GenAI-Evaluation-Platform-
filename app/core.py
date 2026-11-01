"""Synthetic claims workflow with hybrid retrieval, deterministic rules and an audit trail."""
from __future__ import annotations
import json, os, re, sqlite3, time, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TypedDict
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[1]
POLICIES = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
CLAIMS = [
 ("CLM-1001","accidental",12000,120,1,1,1), ("CLM-1002","cosmetic",1500,30,1,1,1),
 ("CLM-1003","accidental",8000,50,1,0,1), ("CLM-1004","accidental",62000,90,1,1,1),
 ("CLM-1005","accidental",9000,410,1,1,1),
]
SCHEMA = """
CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, incident_type TEXT NOT NULL, amount INTEGER NOT NULL,
 days_since_purchase INTEGER NOT NULL, active_coverage INTEGER NOT NULL, proof_of_purchase INTEGER NOT NULL,
 incident_evidence INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, claim_id TEXT NOT NULL,
 question TEXT NOT NULL, status TEXT NOT NULL, reasons_json TEXT NOT NULL, retrieved_ids_json TEXT NOT NULL,
 citations_json TEXT NOT NULL, human_review INTEGER NOT NULL, latency_ms REAL NOT NULL);
"""
@contextmanager
def database():
    path = os.getenv("CLAIMASSIST_DB") or str(ROOT/"data"/"claimassist.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db=sqlite3.connect(path, timeout=10); db.row_factory=sqlite3.Row
    try:
        db.executescript(SCHEMA)
        db.executemany("INSERT OR IGNORE INTO claims VALUES (?,?,?,?,?,?,?)", CLAIMS); db.commit()
        yield db; db.commit()
    except Exception:
        db.rollback(); raise
    finally: db.close()

def claim_lookup(claim_id):
    with database() as db:
        row=db.execute("SELECT * FROM claims WHERE id = ?",(claim_id,)).fetchone()
        return dict(row) if row else None

def tokens(text): return set(re.findall(r"[a-z0-9]+",text.lower()))
@lru_cache(maxsize=1)
def _semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("SEMANTIC_MODEL","sentence-transformers/all-MiniLM-L6-v2"))

def retrieve(question,k=3):
    """Reciprocal-rank fusion of lexical and optional semantic rankings."""
    docs=[p["title"]+" "+p["text"] for p in POLICIES]; qt=tokens(question)
    lexical=[(len(qt & tokens(d))/max(1,len(qt)),i) for i,d in enumerate(docs)]
    rankings=[{i:r for r,(_,i) in enumerate(sorted(lexical,key=lambda x:(-x[0],POLICIES[x[1]]["id"])),1)}]
    if os.getenv("SEMANTIC_RETRIEVAL","0")=="1":
        try:
            import numpy as np
            v=_semantic_model().encode([question]+docs,normalize_embeddings=True)
            scores=[(float(np.dot(v[0],v[i+1])),i) for i in range(len(docs))]
            rankings.append({i:r for r,(_,i) in enumerate(sorted(scores,key=lambda x:(-x[0],POLICIES[x[1]]["id"])),1)})
        except (ImportError,OSError,RuntimeError): pass
    fused={}
    for ranking in rankings:
        for i,rank in ranking.items(): fused[i]=fused.get(i,0)+1/(60+rank)
    return [POLICIES[i] for i in sorted(fused,key=lambda i:(-fused[i],POLICIES[i]["id"]))[:max(1,k)]]

def rules(c):
    if c is None: return "not_found",["Claim ID not found in synthetic data."]
    if not c["proof_of_purchase"] or not c["incident_evidence"]: return "needs_human_review",["Required supporting evidence is missing."]
    if not c["active_coverage"]: return "excluded_by_sample_rules",["Coverage is inactive under sample rules."]
    if c["days_since_purchase"]>365: return "excluded_by_sample_rules",["Incident is outside the sample 365-day window."]
    if c["incident_type"] in {"intentional","cosmetic"}: return "excluded_by_sample_rules",["Sample policy excludes this incident type."]
    if c["amount"]>50000: return "needs_human_review",["Amount exceeds the sample limit and requires manual assessment."]
    return "potentially_eligible",["Sample checks passed; this is not an approval."]

class State(TypedDict,total=False):
    claim_id:str; question:str; prompt_variant:str; evidence:list; claim:dict|None
    status:str; reasons:list; answer:str; citations:list; human_review:bool
    tool_success:bool; verification_passed:bool

def retrieval_node(s): return {"evidence":retrieve(s["question"],3)}
def tools_node(s):
    c=claim_lookup(s["claim_id"]); status,reasons=rules(c)
    return {"claim":c,"status":status,"reasons":reasons,"tool_success":c is not None}
def compose_node(s):
    ids=[p["id"] for p in s.get("evidence",[])]
    answer=(f"Advisory sample-rule status: {s['status']}. {' '.join(s['reasons'])} "
            "A human reviewer must make the final resolution. Retrieved policy references: "+
            (", ".join(f"[{x}]" for x in ids) if ids else "none")+".")
    if os.getenv("ENABLE_LLM","0")=="1" and os.getenv("OPENAI_API_KEY"):
        try:
            import httpx
            context="\n".join(f"{p['id']}: {p['title']} — {p['text']}" for p in s["evidence"])
            r=httpx.post(os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/")+"/chat/completions",
              headers={"Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]},
              json={"model":os.getenv("OPENAI_MODEL","gpt-4o-mini"),"temperature":0,
               "messages":[{"role":"system","content":"Use only supplied evidence. Cite only supplied policy IDs. Never change tool status or claim approval."},
                {"role":"user","content":f"Question: {s['question']}\nStatus: {s['status']}\nReasons: {s['reasons']}\nEvidence:\n{context}"}]},timeout=20)
            r.raise_for_status(); candidate=r.json()["choices"][0]["message"]["content"]
            used=re.findall(r"\[(POL-\d+)\]",candidate); allowed={p["id"] for p in s["evidence"]}
            if used and set(used)<=allowed and s["status"] in candidate:
                answer=candidate+"\nHuman review is mandatory; this is not a final decision."
        except Exception: pass
    return {"answer":answer}
def verify_node(s):
    cited=re.findall(r"\[(POL-\d+)\]",s.get("answer",""))
    allowed={p["id"] for p in s.get("evidence",[])}
    ok=bool(cited) and set(cited)<=allowed
    return {"answer":s["answer"] if ok else "Insufficient verifiable policy evidence. Human review required.",
            "citations":list(dict.fromkeys(cited)) if ok else [],
            "verification_passed":ok,"human_review":True}

builder=StateGraph(State)
for name,fn in [("retrieve",retrieval_node),("tools",tools_node),("compose",compose_node),("verify",verify_node)]: builder.add_node(name,fn)
builder.set_entry_point("retrieve"); builder.add_edge("retrieve","tools"); builder.add_edge("tools","compose")
builder.add_edge("compose","verify"); builder.add_edge("verify",END); graph=builder.compile()

def ask(claim_id,question,prompt_variant="strict"):
    start=time.perf_counter()
    s=graph.invoke({"claim_id":claim_id,"question":question,"prompt_variant":prompt_variant})
    ms=round((time.perf_counter()-start)*1000,2)
    out={"claim_id":claim_id,"question":question,"status":s["status"],"reasons":s["reasons"],
      "answer":s["answer"],"citations":s.get("citations",[]),"human_review":True,
      "tool_success":s["tool_success"],"verification_passed":s.get("verification_passed",False),
      "retrieved_ids":[p["id"] for p in s["evidence"]],"latency_ms":ms}
    with database() as db:
        db.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?,?,?)",
          (str(uuid.uuid4()),datetime.now(timezone.utc).isoformat(),claim_id,question,out["status"],
           json.dumps(out["reasons"]),json.dumps(out["retrieved_ids"]),json.dumps(out["citations"]),1,ms))
    return out

def recent_audit(limit=50):
    with database() as db:
        rows=db.execute("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",(max(1,min(int(limit),200)),)).fetchall()
    result=[]
    for row in rows:
        item=dict(row)
        for key in ("reasons_json","retrieved_ids_json","citations_json"): item[key.removesuffix("_json")]=json.loads(item.pop(key))
        item["human_review"]=bool(item["human_review"]); result.append(item)
    return result
