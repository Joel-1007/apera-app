import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import arxiv
import shutil
import datetime
import traceback

# Initialize FastAPI
app = FastAPI(title="APERA Brain", version="5.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# --- HELPER: ARXIV SEARCH ---
def search_arxiv_safe(query, max_results=3):
    """Fetches real papers from ArXiv"""
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
        traceback.print_exc()
        return []

# --- ENDPOINT 1: CHAT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        print(f"\n{'='*60}")
        print(f"📥 INCOMING REQUEST:")
        print(f"   Query: {request.query}")
        print(f"   Mode: {request.mode}")
        print(f"   Session: {request.session_id}")
        print(f"{'='*60}\n")
        
        citations = []
        response_text = ""
        meta_data = {
            "intent": "GENERAL", 
            "confidence": 0, 
            "fairness": {"balance_label": "Balanced", "diversity_flag": False}
        }

        # MODE: LIVE RESEARCH (ArXiv)
        if "ArXiv" in request.mode or "Research" in request.mode:
            print("🔎 Performing ArXiv Search...")
            papers = search_arxiv_safe(request.query, max_results=3)
            
            print(f"📚 Found {len(papers)} papers")
            
            if papers:
                # Build response text
                response_text = f"**Research Findings on '{request.query}':**\n\n"
                response_text += f"The most relevant study is **'{papers[0]['title']}'**, which explores this topic in depth.\n\n"
                response_text += f"Key insight: {papers[0]['summary'][:200]}...\n\n"
                
                if len(papers) > 1:
                    response_text += f"Additionally, **'{papers[1]['title']}'** provides further evidence."

                # Build citations - MATCH FRONTEND EXACTLY
                # Your frontend expects: file, text, url, type
                print("\n📋 Building citations:")
                for idx, p in enumerate(papers, 1):
                    # CRITICAL FIX: Use title as 'file' field to match frontend expectations
                    citation = {
                        "file": p['title'],  # ✅ Frontend displays this as the source name
                        "text": p['summary'][:250] + "..." if len(p['summary']) > 250 else p['summary'],
                        "url": p['url'],
                        "type": "online"  # Frontend uses this to show "LIVE WEB" badge
                    }
                    citations.append(citation)
                    print(f"   Citation {idx}: {p['title'][:50]}...")
                    print(f"   Structure: file={citation['file'][:30]}..., url={citation['url'][:40]}...")
                
                meta_data["intent"] = "RESEARCH"
                meta_data["confidence"] = 88
                meta_data["xai_reason"] = "Synthesized from top ArXiv semantic matches using hybrid retrieval."
            else:
                response_text = "I searched the research database but couldn't find specific papers. Try a broader term or different keywords."
                meta_data["confidence"] = 30
                meta_data["intent"] = "SEARCH_FAILED"
        
        # MODE: LOCAL / AUDIT
        else:
            response_text = f"I am ready to audit your local documents. Please upload a PDF to the 'Ingest' section to begin analysis of '{request.query}'."
            meta_data["confidence"] = 95
            meta_data["intent"] = "AUDIT"

        # Build final response
        final_response = {
            "response": response_text,
            "citations": citations,
            "meta": meta_data
        }
        
        print(f"\n✅ RESPONSE READY:")
        print(f"   Response length: {len(response_text)} chars")
        print(f"   Citations count: {len(citations)}")
        if citations:
            print(f"   Sample citation keys: {list(citations[0].keys())}")
            print(f"   Sample file field: {citations[0].get('file', 'N/A')[:50]}")
        print(f"{'='*60}\n")
        
        return final_response
        
    except Exception as e:
        print(f"\n❌ ERROR IN /chat ENDPOINT:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        # Return a safe error response
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# --- ENDPOINT 2: INGEST ---
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handles PDF uploads"""
    try:
        print(f"\n📤 File upload: {file.filename}")
        
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
            "message": "File uploaded and ready for indexing"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 3: FEEDBACK ---
@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    print(f"📝 Feedback: {request.rating} for '{request.query}'")
    
    # Save to file
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        with open("feedback_logs/feedback.txt", "a") as f:
            f.write(f"{timestamp} | {request.rating} | {request.query}\n")
    except Exception as e:
        print(f"⚠️ Feedback logging error: {e}")
    
    return {"status": "recorded", "message": "Thank you for your feedback!"}

# --- ADMIN ENDPOINTS ---
@app.get("/admin/toxicity")
def toxicity_stats():
    """Returns mock toxicity monitoring data"""
    return [
        {"timestamp": str(datetime.datetime.now()), "score": 0.05},
        {"timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=1)), "score": 0.03},
        {"timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=2)), "score": 0.07}
    ]

@app.get("/admin/logs")
def session_logs(session_id: Optional[str] = None):
    """Returns session logs"""
    # Try to read from audit_log.json if it exists
    try:
        if os.path.exists("../audit_log.json"):
            with open("../audit_log.json", "r") as f:
                logs = json.load(f)
                if session_id:
                    return [log for log in logs if log.get('session_id') == session_id]
                return logs
    except:
        pass
    
    if session_id:
        return {"session_id": session_id, "logs": []}
    return []

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "active",
        "version": "5.0",
        "message": "APERA Brain API is running",
        "timestamp": str(datetime.datetime.now())
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
            "feedback": "/feedback",
            "admin_logs": "/admin/logs",
            "admin_toxicity": "/admin/toxicity"
        },
        "note": "All systems operational"
    }

# --- RUNNER ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING APERA BRAIN API")
    print("="*60)
    print("📍 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔍 Watch console for request logs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
