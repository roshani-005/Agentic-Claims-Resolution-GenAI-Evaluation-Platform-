"""FastAPI service for synthetic claims review."""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from app.core import ask, claim_lookup, recent_audit
from app.evaluate import evaluate
from app.ocr import extract_image_bytes
app=FastAPI(title="ClaimAssist",version="1.0.0",description="Synthetic claims advisory and GenAI evaluation platform. Not for real decisions.")
class AskRequest(BaseModel):
    claim_id:str=Field(min_length=1,max_length=64,pattern=r"^[A-Za-z0-9-]+$")
    question:str=Field(min_length=3,max_length=2000)
    prompt_variant:str=Field(default="strict",pattern="^(strict|concise)$")
@app.get("/health")
def health():return {"status":"ok","synthetic_data":True,"decision_mode":"advisory_only"}
@app.get("/claims/{claim_id}")
def get_claim(claim_id:str):
    result=claim_lookup(claim_id)
    if result is None:raise HTTPException(404,"Claim not found in synthetic dataset")
    return result
@app.post("/ask")
def ask_endpoint(request:AskRequest):
    try:return ask(request.claim_id,request.question,request.prompt_variant)
    except Exception as exc:raise HTTPException(500,"Claim workflow failed; inspect server logs.") from exc
@app.post("/documents/extract")
async def extract_document(file:UploadFile=File(...),include_tables:bool=Query(True)):
    if file.content_type not in {"image/png","image/jpeg","image/bmp","image/tiff"}:raise HTTPException(415,"Upload PNG, JPG, BMP or TIFF.")
    content=await file.read()
    if len(content)>10*1024*1024:raise HTTPException(413,"Maximum upload size is 10 MB.")
    try:return extract_image_bytes(content,file.filename or "upload.png",include_tables)
    except RuntimeError as exc:raise HTTPException(503,str(exc)) from exc
    except ValueError as exc:raise HTTPException(400,str(exc)) from exc
@app.get("/audit")
def audit(limit:int=Query(50,ge=1,le=200)):return {"items":recent_audit(limit)}
@app.get("/evaluate")
def evaluation():return evaluate()
