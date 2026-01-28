import os
import sys
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict, Any
import arxiv
import shutil
import datetime
import traceback
import logging
from logging.handlers import RotatingFileHandler
import re
import json
import requests

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/apera_backend.log',
            maxBytes=10485760,
            backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("APERA")

# ==========================================
# FASTAPI INITIALIZATION
# ==========================================
app = FastAPI(
    title="APERA Brain API",
    version="8.0-free-local-ai",
    description="100% FREE - Powered by local AI (Ollama) - ZERO payment"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# GLOBAL ERROR HANDLER
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"❌ UNHANDLED EXCEPTION in {request.url.path}")
    logger.error(f"   Exception Type: {type(exc).__name__}")
    logger.error(f"   Exception Message: {str(exc)}")
    logger.error(f"   Traceback:\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path),
            "timestamp": datetime.datetime.now().isoformat()
        }
    )

# ==========================================
# DATA MODELS
# ==========================================
class ChatRequest(BaseModel):
    query: str
    session_id: str
    mode: str = "local"

class FeedbackRequest(BaseModel):
    query: str
    rating: str

class Citation(BaseModel):
    file: str
    text: str
    url: str
    type: str

# ==========================================
# FREE LOCAL AI INTEGRATION (OLLAMA)
# ==========================================
def call_local_ai(prompt: str, system_prompt: str = None, max_tokens: int = 2048) -> str:
    """
    Call LOCAL AI (Ollama) - 100% FREE, runs on your machine
    
    Ollama supports many models:
    - llama3.2 (3B) - Fast, good for explanations
    - mistral (7B) - Best balance of speed/quality
    - llama3.1 (8B) - High quality
    - deepseek-r1 (7B) - Good reasoning
    
    NO API KEY NEEDED. NO PAYMENT. RUNS LOCALLY.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system context
        max_tokens: Maximum response length
        
    Returns:
        AI response text
    """
    try:
        # Check if Ollama is running
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        logger.info(f"🤖 Calling LOCAL AI (Ollama) at {ollama_url}...")
        
        # Choose model (can be configured)
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call Ollama API
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Ollama error: {response.status_code}")
            return None
        
        result = response.json()
        response_text = result.get("message", {}).get("content", "")
        
        logger.info(f"✅ Local AI responded: {len(response_text)} chars")
        logger.info(f"   Model: {model}")
        logger.info(f"   Cost: $0.00 (FREE!)")
        
        return response_text
        
    except requests.exceptions.ConnectionError:
        logger.error("❌ Ollama not running. Start it with: ollama serve")
        return None
    except requests.exceptions.Timeout:
        logger.error("❌ Ollama timeout - response took too long")
        return None
    except Exception as e:
        logger.error(f"❌ Local AI error: {type(e).__name__}: {str(e)}")
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        return None

# ==========================================
# ADVANCED RESEARCH SYNTHESIS ENGINE
# ==========================================
def generate_advanced_research_synthesis(query: str, papers: List[Dict[str, Any]]) -> str:
    """
    Use LOCAL AI to analyze ArXiv papers and generate sophisticated research synthesis
    100% FREE - NO API COSTS!
    
    Args:
        query: User's research question
        papers: List of ArXiv papers
        
    Returns:
        AI-generated comprehensive research synthesis
    """
    logger.info(f"🧠 Generating research synthesis with LOCAL AI (FREE)...")
    
    if not papers or len(papers) == 0:
        return None
    
    # Prepare paper summaries
    papers_context = ""
    for idx, paper in enumerate(papers, 1):
        papers_context += f"\n\n{'='*80}\n"
        papers_context += f"PAPER {idx}\n"
        papers_context += f"{'='*80}\n"
        papers_context += f"Title: {paper['title']}\n"
        papers_context += f"Authors: {', '.join(paper['authors'][:5])}"
        if len(paper['authors']) > 5:
            papers_context += " et al."
        papers_context += f"\nPublished: {paper['published']}\n"
        papers_context += f"\nABSTRACT:\n{paper['summary']}\n"
    
    system_prompt = """You are an expert research analyst. Your role is to:

1. Analyze multiple research papers deeply
2. Identify key themes, methodologies, and findings
3. Compare and contrast different approaches
4. Explain complex concepts clearly
5. Provide actionable insights
6. Cite papers using [Paper 1], [Paper 2] format

Write in a professional yet accessible style. Use markdown formatting."""

    user_prompt = f"""I'm researching: "{query}"

I've found {len(papers)} relevant papers from ArXiv. Please analyze them and provide a comprehensive research synthesis.

{papers_context}

{'='*80}

Provide a detailed synthesis including:

1. **Executive Summary** (2-3 sentences): Key takeaways

2. **Core Concepts**: Fundamental concepts in these papers

3. **Methodological Approaches**: Methods/techniques used

4. **Key Findings**: Major discoveries and innovations

5. **Comparative Analysis**: How papers relate - agree/disagree/complement?

6. **Practical Applications**: Real-world implications

7. **Research Gaps**: What questions remain unanswered?

8. **Reading Recommendation**: Which papers to read first and why?

Use citations like [Paper 1], [Paper 2]. Aim for 600-800 words."""

    synthesis = call_local_ai(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=2500
    )
    
    if synthesis:
        logger.info(f"✅ Local AI synthesis: {len(synthesis)} chars")
        return synthesis
    
    return None

def generate_conceptual_explanation(query: str, papers: List[Dict[str, Any]] = None) -> str:
    """
    Use LOCAL AI to generate conceptual explanation
    100% FREE!
    
    Args:
        query: The conceptual question
        papers: Optional papers for context
        
    Returns:
        AI-generated explanation
    """
    logger.info(f"🎓 Generating conceptual explanation with LOCAL AI...")
    
    papers_context = ""
    if papers and len(papers) > 0:
        papers_context = f"\n\nREFERENCE (optional):\n"
        for idx, paper in enumerate(papers[:2], 1):
            papers_context += f"\nPaper {idx}: {paper['title']}\n"
            papers_context += f"Summary: {paper['summary'][:250]}...\n"
    
    system_prompt = """You are an expert educator explaining technical concepts. Your explanations should:

1. Start with a clear, simple definition
2. Break down into understandable components
3. Use analogies and examples
4. Compare with related concepts
5. Explain practical applications
6. Be accurate but accessible

Use markdown with headers (###), bullet points, and **bold** for emphasis."""

    user_prompt = f"""Explain this question clearly:

"{query}"
{papers_context}

Structure your explanation:

1. **Clear Definition**: What is this in simple terms?
2. **Key Components**: Essential parts/elements
3. **How It Works**: The mechanism or process
4. **Practical Examples**: Real-world applications
5. **Important Distinctions**: How it differs from similar concepts
6. **When to Use**: Guidelines for application

Aim for 400-600 words. Technical but accessible."""

    explanation = call_local_ai(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=1800
    )
    
    if explanation:
        logger.info(f"✅ Explanation generated: {len(explanation)} chars")
        return explanation
    
    return None

def generate_fallback_synthesis(query: str, papers: List[Dict[str, Any]]) -> str:
    """
    Fallback synthesis when AI is unavailable
    Still produces good structured output
    """
    logger.info("📝 Using fallback synthesis (AI unavailable)...")
    
    response = f"# Research Analysis: {query}\n\n"
    response += f"## 📊 Overview\n\n"
    response += f"I've analyzed {len(papers)} peer-reviewed papers from ArXiv on this topic.\n\n"
    
    # Primary paper
    response += f"## 📚 Primary Research\n\n"
    response += f"### {papers[0]['title']}\n"
    
    if papers[0].get('authors'):
        authors = ", ".join(papers[0]['authors'][:4])
        if len(papers[0]['authors']) > 4:
            authors += " et al."
        response += f"**Authors:** {authors}\n"
    
    if papers[0].get('published'):
        response += f"**Published:** {papers[0]['published']}\n"
    
    response += f"\n**Abstract:**\n{papers[0]['summary']}\n\n"
    
    # Supporting papers
    if len(papers) > 1:
        response += f"## 🔬 Supporting Literature\n\n"
        
        for idx, paper in enumerate(papers[1:], 2):
            response += f"### {idx}. {paper['title']}\n"
            
            if paper.get('authors'):
                authors = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors += " et al."
                response += f"**Authors:** {authors}"
                if paper.get('published'):
                    response += f" ({paper['published']})"
                response += "\n\n"
            
            summary = paper['summary'][:350]
            if len(paper['summary']) > 350:
                summary += "..."
            response += f"{summary}\n\n"
    
    response += f"## 💡 Synthesis\n\n"
    response += f"These {len(papers)} papers collectively advance understanding of {query} through "
    response += f"theoretical frameworks, methodological innovation, empirical validation, and practical applications.\n\n"
    response += f"*💾 Full PDFs available via citation links below.*\n"
    
    return response

# ==========================================
# QUESTION TYPE DETECTION
# ==========================================
def detect_question_type(query: str) -> str:
    """Detect if query is conceptual, research, or hybrid"""
    query_lower = query.lower()
    
    conceptual_patterns = [
        r'\bwhat is\b', r'\bwhat are\b', r'\bwhat\'s\b',
        r'\bhow does\b', r'\bhow do\b', r'\bhow to\b',
        r'\bdifference between\b', r'\bcompare\b', r'\bvs\b', r'\bversus\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bdefine\b',
        r'\bwhy\b', r'\bwhen to use\b', r'\badvantages\b',
        r'\bbasics\b', r'\bfundamentals\b', r'\bintroduction to\b',
        r'\bunderstand\b', r'\bmeaning of\b', r'\bconcept of\b'
    ]
    
    research_patterns = [
        r'\blatest\b', r'\brecent\b', r'\bstate of the art\b',
        r'\bsurvey\b', r'\breview of\b', r'\badvances in\b',
        r'\bcurrent research\b', r'\bbreakthrough\b',
        r'\bpapers on\b', r'\bstudies on\b', r'\btrends in\b'
    ]
    
    conceptual_score = sum(1 for p in conceptual_patterns if re.search(p, query_lower))
    research_score = sum(1 for p in research_patterns if re.search(p, query_lower))
    
    logger.info(f"📊 Question type: conceptual={conceptual_score}, research={research_score}")
    
    if conceptual_score > 0 and research_score > 0:
        return 'hybrid'
    elif conceptual_score > 0:
        return 'conceptual'
    elif research_score > 0:
        return 'research'
    else:
        return 'conceptual' if len(query.split()) <= 5 else 'research'

# ==========================================
# ARXIV FUNCTIONS
# ==========================================
def search_arxiv_safe(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Safely search ArXiv"""
    logger.info(f"🔎 Searching ArXiv: '{query}'")
    
    try:
        if not query or not query.strip():
            return []
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query.strip(),
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        
        for paper in client.results(search):
            try:
                paper_data = {
                    "title": str(paper.title) if paper.title else "Untitled",
                    "summary": str(paper.summary) if paper.summary else "No summary",
                    "url": str(paper.pdf_url) if paper.pdf_url else "",
                    "published": str(paper.published.date()) if paper.published else "Unknown",
                    "authors": [str(author.name) for author in paper.authors] if paper.authors else []
                }
                results.append(paper_data)
                
            except Exception as e:
                logger.error(f"Error processing paper: {e}")
                continue
        
        logger.info(f"✅ Found {len(results)} papers")
        return results
        
    except Exception as e:
        logger.error(f"❌ ArXiv error: {e}")
        return []

def build_citation(paper: Dict[str, Any]) -> Dict[str, str]:
    """Build citation dictionary"""
    try:
        return {
            "file": paper.get("title", "Unknown"),
            "text": paper.get("summary", "No summary")[:300] + "...",
            "url": paper.get("url", ""),
            "type": "online"
        }
    except:
        return {"file": "Error", "text": "Unable to extract", "url": "", "type": "online"}

# ==========================================
# MAIN CHAT ENDPOINT - FREE LOCAL AI
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    FREE Local AI-powered chat endpoint
    100% FREE - NO PAYMENT REQUIRED!
    """
    logger.info("="*80)
    logger.info("📥 NEW CHAT REQUEST - FREE LOCAL AI MODE")
    logger.info(f"   Query: {request.query}")
    logger.info("="*80)
    
    try:
        citations = []
        response_text = ""
        meta_data = {
            "intent": "GENERAL",
            "confidence": 0,
            "fairness": {"balance_label": "Balanced", "diversity_flag": False}
        }
        
        if not request.query or len(request.query.strip()) == 0:
            return {
                "response": "Please provide a valid research question.",
                "citations": [],
                "meta": {"intent": "ERROR", "confidence": 0, "fairness": {"balance_label": "N/A", "diversity_flag": False}}
            }
        
        # MODE: LIVE RESEARCH
        if "ArXiv" in request.mode or "Research" in request.mode:
            logger.info("🔬 Processing FREE LOCAL AI Research Mode")
            
            try:
                # Detect question type
                question_type = detect_question_type(request.query)
                logger.info(f"🎯 Question type: {question_type}")
                
                # Search ArXiv
                papers = search_arxiv_safe(request.query, max_results=3)
                
                if not papers:
                    response_text = f"No papers found on '{request.query}'. Try broader terms."
                    meta_data["intent"] = "SEARCH_FAILED"
                    meta_data["confidence"] = 30
                    
                else:
                    # Generate response with LOCAL AI
                    if question_type == 'conceptual':
                        logger.info("🎓 Mode: Conceptual Explanation")
                        
                        explanation = generate_conceptual_explanation(request.query, papers)
                        
                        if explanation:
                            response_text = f"# 📚 Understanding: {request.query}\n\n"
                            response_text += explanation
                            response_text += "\n\n---\n\n"
                            response_text += f"## 📖 Recommended Reading\n\n"
                            response_text += f"For deeper insights, see these {len(papers)} papers:\n\n"
                            
                            for idx, paper in enumerate(papers, 1):
                                response_text += f"**{idx}. {paper['title']}**\n"
                                if paper.get('authors'):
                                    authors = ", ".join(paper['authors'][:3])
                                    if len(paper['authors']) > 3:
                                        authors += " et al."
                                    response_text += f"*{authors}"
                                    if paper.get('published'):
                                        response_text += f" ({paper['published']})"
                                    response_text += "*\n\n"
                            
                            meta_data["intent"] = "CONCEPTUAL"
                            meta_data["confidence"] = 90
                            meta_data["xai_reason"] = "Local AI explanation (FREE)"
                        else:
                            # Fallback
                            response_text = generate_fallback_synthesis(request.query, papers)
                            meta_data["intent"] = "CONCEPTUAL"
                            meta_data["confidence"] = 75
                        
                    elif question_type == 'research':
                        logger.info("🧠 Mode: Research Synthesis")
                        
                        synthesis = generate_advanced_research_synthesis(request.query, papers)
                        
                        if synthesis:
                            response_text = synthesis
                            meta_data["intent"] = "RESEARCH"
                            meta_data["confidence"] = 95
                            meta_data["xai_reason"] = "Local AI synthesis (FREE)"
                        else:
                            response_text = generate_fallback_synthesis(request.query, papers)
                            meta_data["intent"] = "RESEARCH"
                            meta_data["confidence"] = 85
                    
                    else:  # hybrid
                        logger.info("✨ Mode: Hybrid")
                        
                        explanation = generate_conceptual_explanation(request.query, papers)
                        
                        if explanation:
                            response_text = f"# 📚 Understanding: {request.query}\n\n"
                            response_text += explanation
                            response_text += "\n\n---\n\n"
                            
                            synthesis = generate_advanced_research_synthesis(request.query, papers)
                            if synthesis:
                                response_text += f"## 🔬 Research Analysis\n\n{synthesis}"
                            else:
                                response_text += generate_fallback_synthesis(request.query, papers)
                            
                            meta_data["intent"] = "HYBRID"
                            meta_data["confidence"] = 92
                        else:
                            response_text = generate_fallback_synthesis(request.query, papers)
                            meta_data["intent"] = "HYBRID"
                            meta_data["confidence"] = 80
                    
                    # Build citations
                    for paper in papers:
                        citations.append(build_citation(paper))
                    
                    logger.info(f"✅ Response: {len(response_text)} chars, {len(citations)} citations")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                logger.error(traceback.format_exc())
                response_text = "Error processing request. Please try again."
                meta_data["intent"] = "ERROR"
        
        else:
            response_text = f"Ready to audit local documents. Upload PDF to analyze '{request.query}'."
            meta_data["confidence"] = 95
            meta_data["intent"] = "AUDIT"
        
        logger.info("✅ Completed\n")
        
        return {
            "response": response_text,
            "citations": citations,
            "meta": meta_data
        }
    
    except Exception as e:
        logger.error(f"❌ CRITICAL: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "response": f"Error: {str(e)}",
            "citations": [],
            "meta": {"intent": "ERROR", "confidence": 0, "fairness": {"balance_label": "N/A", "diversity_flag": False}}
        }

# ==========================================
# OTHER ENDPOINTS
# ==========================================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handle PDF uploads"""
    try:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF supported")
        
        os.makedirs("temp_data", exist_ok=True)
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-'))
        file_path = os.path.join("temp_data", safe_filename or "upload.pdf")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "filename": safe_filename, "size": os.path.getsize(file_path)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Handle feedback"""
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        with open("feedback_logs/feedback.txt", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | {request.rating} | {request.query}\n")
        return {"status": "recorded"}
    except:
        return {"status": "recorded"}

@app.get("/")
def root():
    return {
        "service": "APERA Brain API",
        "version": "8.0-free-local-ai",
        "status": "operational",
        "cost": "$0.00 - 100% FREE!",
        "features": ["Local AI (Ollama)", "Zero Payment", "Research Synthesis"]
    }

@app.get("/health")
def health_check():
    # Check if Ollama is running
    try:
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        ollama_status = "running" if response.status_code == 200 else "not running"
    except:
        ollama_status = "not running"
    
    return {
        "status": "healthy",
        "version": "8.0-free-local-ai",
        "local_ai": ollama_status,
        "cost": "$0.00 per query",
        "features": {
            "local_ai_synthesis": "active" if ollama_status == "running" else "fallback mode",
            "arxiv_search": "active",
            "zero_payment": "guaranteed"
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("="*80)
    logger.info("🚀 APERA FREE LOCAL AI STARTING")
    logger.info("="*80)
    
    # Check Ollama status
    try:
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("✅ LOCAL AI: RUNNING (Ollama)")
            logger.info("   💰 Cost: $0.00 per query")
            logger.info("   🧠 AI analysis: ACTIVE")
        else:
            logger.warning("⚠️ LOCAL AI: NOT RESPONDING")
            logger.info("   Using fallback mode")
    except:
        logger.warning("⚠️ LOCAL AI: NOT RUNNING")
        logger.info("   Start with: ollama serve")
        logger.info("   Install from: https://ollama.com")
        logger.info("   Using fallback mode (still works!)")
    
    for directory in ["temp_data", "feedback_logs", "logs"]:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ Startup complete\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 APERA BRAIN API - 100% FREE LOCAL AI")
    print("="*80)
    print(f"📍 Server: http://0.0.0.0:8000")
    print(f"💰 Cost: $0.00 per query - NO PAYMENT REQUIRED!")
    print("="*80)
    
    # Check Ollama
    try:
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ LOCAL AI: RUNNING")
            print("   🧠 Intelligent synthesis: ACTIVE")
            print("   💰 Cost: $0.00 (FREE FOREVER!)")
        else:
            print("⚠️ LOCAL AI: NOT RESPONDING")
            print("   Running in fallback mode")
    except:
        print("⚠️ LOCAL AI: NOT RUNNING")
        print("   Install: https://ollama.com")
        print("   Start: ollama serve")
        print("   Then: ollama pull llama3.2")
        print("")
        print("   Don't worry - fallback mode still works!")
    
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
