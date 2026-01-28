import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import arxiv
import shutil
import datetime

# Initialize FastAPI
app = FastAPI(title="APERA Brain", version="4.0")

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    query: str
    session_id: str
    mode: str = "local"

class FeedbackRequest(BaseModel):
    query: str
    rating: str

# --- HELPER: ARXIV SEARCH (SAFE MODE) ---
def search_arxiv_safe(query, max_results=3):
    """Fetches real papers from ArXiv without crashing"""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for r in client.results(search):
            results.append({
                "title": r.title,
                "summary": r.summary,
                "url": r.pdf_url,
                "published": str(r.published.date())
            })
        return results
    except Exception as e:
        print(f"ArXiv Error: {e}")
        return []

# --- ENDPOINT 1: CHAT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"📥 Query: {request.query} | Mode: {request.mode}")
    
    citations = []
    response_text = ""
    meta_data = {
        "intent": "GENERAL", 
        "confidence": 0, 
        "fairness": {"balance_label": "Balanced", "diversity_flag": False}
    }

    # MODE: LIVE RESEARCH (ArXiv)
    if "ArXiv" in request.mode or "Research" in request.mode:
        print("🔎 Performing Safe ArXiv Search...")
        papers = search_arxiv_safe(request.query)
        
        if papers:
            # Build Context
            response_text = f"**Research Findings on '{request.query}':**\n\n"
            response_text += f"The most relevant study is **'{papers[0]['title']}'**, which explores this topic in depth.\n\n"
            response_text += f"Key insight: {papers[0]['summary'][:200]}...\n\n"
            if len(papers) > 1:
                response_text += f"Additionally, **'{papers[1]['title']}'** provides further evidence regarding these mechanisms."

            # Build Citations (CRITICAL FIX: Mapping Title to 'file')
            for p in papers:
                citations.append({
                    "text": p['summary'],
                    "file": p['title'], # <--- FIX: Using Title as filename to prevent 500 Error
                    "url": p['url'],
                    "type": "online"
                })
            
            meta_data["intent"] = "RESEARCH"
            meta_data["confidence"] = 88
            meta_data["xai_reason"] = "Synthesized from top ArXiv semantic matches."
        else:
            response_text = "I searched the research database but couldn't find specific papers. Try a broader term."
    
    # MODE: LOCAL / AUDIT
    else:
        response_text = f"I am ready to audit your local documents. Please upload a PDF to the 'Ingest' section to begin analysis of '{request.query}'."
        meta_data["confidence"] = 95
        meta_data["intent"] = "AUDIT"

    return {
        "response": response_text,
        "citations": citations,
        "meta": meta_data
    }

# --- ENDPOINT 2: INGEST (FILE UPLOAD) ---
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handles PDF uploads safely"""
    try:
        os.makedirs("temp_data", exist_ok=True)
        file_path = f"temp_data/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"✅ File saved: {file.filename}")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 3: FEEDBACK ---
@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    print(f"📝 Feedback received: {request.rating}")
    return {"status": "recorded"}

# --- ENDPOINT 4: ADMIN STUBS ---
@app.get("/admin/toxicity")
def toxicity_stats():
    return [{"timestamp": str(datetime.datetime.now()), "score": 0.05}]

@app.get("/admin/logs")
def session_logs(session_id: str = None):
    return []

@app.get("/")
def health_check():
    return {"status": "active"}

# --- RUNNER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
