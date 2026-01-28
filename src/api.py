import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import arxiv
import shutil
import datetime

# Initialize FastAPI
app = FastAPI(title="APERA Brain", version="5.0")

# Add CORS middleware to prevent frontend connection issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                "published": str(r.published.date()),
                "authors": [author.name for author in r.authors]
            })
        return results
    except Exception as e:
        print(f"❌ ArXiv Error: {e}")
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

            # Build Citations - FIXED FORMAT
            # The frontend expects citations without 'file' field for online sources
            for idx, p in enumerate(papers, 1):
                citation = {
                    "text": p['summary'][:300] + "..." if len(p['summary']) > 300 else p['summary'],
                    "title": p['title'],
                    "url": p['url'],
                    "type": "online",
                    "source": f"ArXiv - {p['published']}"
                }
                
                # Add authors if available
                if p.get('authors'):
                    citation["authors"] = ", ".join(p['authors'][:3])  # First 3 authors
                
                citations.append(citation)
            
            meta_data["intent"] = "RESEARCH"
            meta_data["confidence"] = 88
            meta_data["xai_reason"] = "Synthesized from top ArXiv semantic matches."
        else:
            response_text = "I searched the research database but couldn't find specific papers. Try a broader term or different keywords."
            meta_data["confidence"] = 30
    
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
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        os.makedirs("temp_data", exist_ok=True)
        
        # Sanitize filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-'))
        file_path = f"temp_data/{safe_filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"✅ File saved: {safe_filename}")
        return {
            "status": "success", 
            "filename": safe_filename,
            "message": "File uploaded successfully and ready for analysis"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# --- ENDPOINT 3: FEEDBACK ---
@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    print(f"📝 Feedback received: {request.rating} for query: {request.query}")
    
    # Save feedback to file (optional)
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        with open("feedback_logs/feedback.txt", "a") as f:
            f.write(f"{timestamp} | {request.rating} | {request.query}\n")
    except Exception as e:
        print(f"⚠️ Feedback logging error: {e}")
    
    return {"status": "recorded", "message": "Thank you for your feedback!"}

# --- ENDPOINT 4: ADMIN STUBS ---
@app.get("/admin/toxicity")
def toxicity_stats():
    """Returns mock toxicity monitoring data"""
    return [
        {"timestamp": str(datetime.datetime.now()), "score": 0.05},
        {"timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=1)), "score": 0.03}
    ]

@app.get("/admin/logs")
def session_logs(session_id: Optional[str] = None):
    """Returns session logs (stub)"""
    if session_id:
        return {"session_id": session_id, "logs": []}
    return {"logs": []}

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "active",
        "version": "5.0",
        "message": "APERA Brain API is running"
    }

@app.get("/health")
def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": str(datetime.datetime.now()),
        "endpoints": {
            "chat": "/chat",
            "ingest": "/ingest",
            "feedback": "/feedback"
        }
    }

# --- RUNNER ---
if __name__ == "__main__":
    print("🚀 Starting APERA Brain API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
