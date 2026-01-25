import sys
import os
from fastapi import UploadFile, File
import shutil
import os

# --- CRITICAL FIX: Add Project Root to Path ---
# This forces Python to see 'src' as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from src.agent_guarded import RAGSystem
from src.database import init_db, log_interaction, update_feedback, get_toxicity_trends, get_session_logs

# 1. Initialize App & DB
app = FastAPI(title="APERA AI Engine", version="3.0")
Instrumentator().instrument(app).expose(app)
init_db()

# 2. Initialize Brain
print("🚀 Booting RAG Engine with Fairlearn & Semantic Scholar...")
agent = RAGSystem()

class QueryRequest(BaseModel):
    query: str
    session_id: str
    mode: str = "local"

class FeedbackRequest(BaseModel):
    query: str
    rating: str

@app.get("/")
def health_check():
    return {"status": "active", "monitoring": "prometheus_enabled"}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # Run Agent
        result = agent.chat(request.query, mode=request.mode)
        
        # Log Logic
        is_safe = True
        if "Blocked" in result['response']: is_safe = False
        
        log_interaction(request.session_id, request.query, result['response'], result['meta'], {'is_safe': is_safe})
        
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    update_feedback(request.query, request.rating)
    return {"status": "recorded"}

@app.get("/admin/toxicity")
def toxicity_stats():
    return get_toxicity_trends()

@app.get("/admin/logs")
def session_logs(session_id: str = None):
    return get_session_logs(session_id)

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # Create temp folder
        os.makedirs("temp_data", exist_ok=True)
        file_path = f"temp_data/{file.filename}"
        
        # Save file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Add to RAG system (if your rag object supports it)
        if hasattr(agent, 'ingest'):
             agent.ingest(file_path)
        
        return {"message": "File processed", "filename": file.filename}
    except Exception as e:
        print(f"Error: {e}")
        return {"detail": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
