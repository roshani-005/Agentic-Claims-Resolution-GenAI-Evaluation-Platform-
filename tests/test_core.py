from app.core import ask, claim_lookup, rules
from app.evaluate import evaluate
from app.main import app
from fastapi.testclient import TestClient

def test_rules():
    assert rules(claim_lookup("CLM-1001"))[0] == "potentially_eligible"
    assert rules(claim_lookup("CLM-1002"))[0] == "excluded_by_sample_rules"
    assert rules(claim_lookup("CLM-1003"))[0] == "needs_human_review"

def test_citations():
    result = ask("CLM-1001", "Is accidental damage covered?")
    assert result["citations"]
    assert set(result["citations"]).issubset(set(result["retrieved_ids"]))
    assert "human" in result["answer"].lower()

def test_api():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/claims/missing").status_code == 404
    assert client.post("/ask", json={"claim_id":"CLM-1001","question":"Is this eligible?"}).status_code == 200

def test_eval():
    report = evaluate()
    assert report["strict"]["status_accuracy"] == 1
