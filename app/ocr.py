"""Optional PaddleOCR 2.x adapter. OCR results are untrusted and require human verification."""
import re, tempfile
from pathlib import Path

def character_error_rate(reference,hypothesis):
    reference,hypothesis=reference.strip(),hypothesis.strip()
    if not reference:return 0.0 if not hypothesis else 1.0
    prev=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        cur=[i]
        for j,b in enumerate(hypothesis,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(a!=b)))
        prev=cur
    return prev[-1]/len(reference)

def extract_fields(text):
    patterns={"claim_id":r"\b(CLM[- ]?\d{3,})\b","policy_number":r"\b(?:policy|pol)\s*(?:no\.?|number)?\s*[:#-]?\s*([A-Z0-9/-]{4,})\b",
      "amount":r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d{1,2})?)","date":r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"}
    return {k:(m.group(1) if (m:=re.search(p,text,re.I)) else None) for k,p in patterns.items()}

def extract_image_bytes(content,filename="upload.png",include_tables=True):
    if not content:raise ValueError("Uploaded document is empty.")
    if Path(filename).suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:raise ValueError("Supported formats: PNG, JPG, BMP and TIFF.")
    try:from paddleocr import PaddleOCR, PPStructure
    except ImportError as exc:raise RuntimeError("Install optional OCR dependencies with pip install -r requirements-ocr.txt.") from exc
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as f:
        f.write(content);f.flush();raw=PaddleOCR(use_angle_cls=True,lang="en",show_log=False).ocr(f.name,cls=True)
        lines=[{"text":str(x[1][0]),"confidence":float(x[1][1])} for page in (raw or []) for x in (page or []) if x and len(x)>1 and x[1]]
        text="\n".join(x["text"] for x in lines); tables=[]
        if include_tables:
            for block in PPStructure(show_log=False)(f.name):
                if block.get("type")=="table":tables.append(block.get("res",{}).get("html",""))
    return {"text":text,"lines":lines,"fields":extract_fields(text),"tables_html":tables,
      "review_required":True,"engine":"PaddleOCR + PP-Structure"}
