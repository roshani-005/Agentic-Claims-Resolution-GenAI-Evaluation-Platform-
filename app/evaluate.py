"""Repeatable evaluation on a tiny synthetic benchmark, not production validation."""
import json, statistics
from pathlib import Path
from app.core import ask
CASES=json.loads((Path(__file__).resolve().parents[1]/"data/eval_cases.json").read_text(encoding="utf-8"))
def percentile(values,q=95):
    if not values:return 0.0
    vals=sorted(values); return round(vals[min(len(vals)-1,max(0,int((q/100)*(len(vals)-1)+0.999999)))],2)
def evaluate():
    variants={}
    for variant in ("strict","concise"):
        rows=[]
        for case in CASES:
            r=ask(case["claim_id"],case["question"],variant); relevant=set(case["relevant_ids"]); got=set(r["retrieved_ids"]); cites=r["citations"]
            rows.append({"claim_id":case["claim_id"],"status_match":r["status"]==case["expected_status"],
              "retrieval_recall_at_3":len(relevant&got)/max(1,len(relevant)),
              "citation_precision":sum(c in got for c in cites)/max(1,len(cites)),
              "unsupported_citation_rate":sum(c not in got for c in cites)/max(1,len(cites)),
              "human_review":r["human_review"],"verification_passed":r["verification_passed"],
              "tool_success":r["tool_success"],"latency_ms":r["latency_ms"]})
        lat=[x["latency_ms"] for x in rows]
        variants[variant]={"case_count":len(rows),
          "status_accuracy":round(statistics.mean(x["status_match"] for x in rows),4),
          "retrieval_recall_at_3":round(statistics.mean(x["retrieval_recall_at_3"] for x in rows),4),
          "citation_precision":round(statistics.mean(x["citation_precision"] for x in rows),4),
          "unsupported_citation_rate":round(statistics.mean(x["unsupported_citation_rate"] for x in rows),4),
          "human_review_rate":round(statistics.mean(x["human_review"] for x in rows),4),
          "verification_pass_rate":round(statistics.mean(x["verification_passed"] for x in rows),4),
          "tool_success_rate":round(statistics.mean(x["tool_success"] for x in rows),4),
          "mean_latency_ms":round(statistics.mean(lat),2),"p95_latency_ms":percentile(lat),
          "estimated_cost_usd":0.0,"cost_note":"Offline deterministic mode; no provider billing. LLM costs require token telemetry.",
          "cases":rows}
    return {"benchmark":"synthetic_demo_v1","warning":"Tiny synthetic benchmark; not production evidence.","variants":variants}
if __name__=="__main__": print(json.dumps(evaluate(),indent=2))
