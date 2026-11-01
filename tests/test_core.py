import pytest
from fastapi.testclient import TestClient
from app.core import ask,claim_lookup,retrieve,rules
from app.evaluate import evaluate
from app.main import app
from app.ocr import character_error_rate,extract_fields
client=TestClient(app)
def test_rules():
    assert rules(claim_lookup("CLM-1001"))[0]=="potentially_eligible"
    assert rules(claim_lookup("CLM-1002"))[0]=="excluded_by_sample_rules"
    assert rules(claim_lookup("CLM-1003"))[0]=="needs_human_review"
    assert rules(claim_lookup("CLM-1004"))[0]=="needs_human_review"
    assert rules(claim_lookup("CLM-1005"))[0]=="excluded_by_sample_rules"
    assert rules(None)[0]=="not_found"
def test_retrieval_does_not_inject_policies():
    result=retrieve("cosmetic scratches and exclusions",3)
    assert 1<=len(result)<=3 and any(p["id"]=="POL-02" for p in result)
def test_citations_and_human_gate():
    r=ask("CLM-1001","Is accidental damage covered?")
    assert r["citations"] and set(r["citations"])<=set(r["retrieved_ids"])
    assert r["human_review"] is True
def test_unknown_claim():
    r=ask("CLM-9999","Check this claim")
    assert r["status"]=="not_found" and not r["tool_success"] and r["human_review"]
def test_api():
    assert client.get("/health").status_code==200
    assert client.get("/claims/missing").status_code==404
    assert client.post("/ask",json={"claim_id":"CLM-1001","question":"Is this eligible?"}).status_code==200
    assert client.get("/audit").status_code==200
    assert client.get("/evaluate").status_code==200
def test_ocr_helpers():
    assert character_error_rate("claim","claim")==0
    assert character_error_rate("claim","clain")==pytest.approx(.2)
    fields=extract_fields("Claim CLM-1001 Amount ₹12,000 Policy No: ABCD1234")
    assert fields["claim_id"]=="CLM-1001" and fields["amount"]=="12,000" and fields["policy_number"]=="ABCD1234"
def test_evaluation_metrics():
    report=evaluate(); strict=report["variants"]["strict"]
    assert strict["case_count"]==3 and 0<=strict["status_accuracy"]<=1
    assert strict["p95_latency_ms"]>=0 and strict["human_review_rate"]==1
