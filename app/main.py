from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.core import ask, claim_lookup
from app.evaluate import evaluate

app = FastAPI(title="ClaimAssist", version="0.1.0", description="Synthetic claims advisory and evaluation platform")

class AskRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=3, max_length=2000)
    prompt_variant: str = Field(default="strict", pattern="^(strict|concise)$")

@app.get("/health")
def health():
    return {"status": "ok", "synthetic_data": True}

@app.get("/claims/{claim_id}")
def get_claim(claim_id: str):
    result = claim_lookup(claim_id)
    if result is None:
        raise HTTPException(404, "Claim not found")
    return result

@app.post("/ask")
def ask_endpoint(request: AskRequest):
    return ask(request.claim_id, request.question, request.prompt_variant)

@app.get("/evaluate")
def evaluation():
    return evaluate()
