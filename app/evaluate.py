"""Fixed, reproducible evaluation. These are heuristic metrics, not independent factuality judgments."""
import json
import statistics
from pathlib import Path
from app.core import ask

CASES = json.loads((Path(__file__).resolve().parents[1] / "data/eval_cases.json").read_text())

def evaluate():
    output = {}
    for variant in ("strict", "concise"):
        rows = []
        for case in CASES:
            result = ask(case["claim_id"], case["question"], variant)
            relevant = set(case["relevant_ids"])
            retrieved = set(result["retrieved_ids"])
            citations = result["citations"]
            rows.append({"claim_id": case["claim_id"], "status_match": result["status"] == case["expected_status"],
                "retrieval_recall_at_3": len(relevant & retrieved) / len(relevant),
                "citation_precision": sum(c in retrieved for c in citations) / len(citations) if citations else 0,
                "unsupported_citation_rate": sum(c not in retrieved for c in citations) / len(citations) if citations else 0,
                "tool_success": result["tool_success"], "latency_ms": result["latency_ms"]})
        output[variant] = {"cases": rows, "status_accuracy": statistics.mean(r["status_match"] for r in rows),
            "retrieval_recall_at_3": statistics.mean(r["retrieval_recall_at_3"] for r in rows),
            "citation_precision": statistics.mean(r["citation_precision"] for r in rows),
            "unsupported_citation_rate": statistics.mean(r["unsupported_citation_rate"] for r in rows),
            "tool_success_rate": statistics.mean(r["tool_success"] for r in rows),
            "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 2)}
    return output

if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
