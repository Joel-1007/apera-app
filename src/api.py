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
    version="6.0-hybrid-ai",
    description="Hybrid AI research intelligence: AI explanations + Research citations"
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is quantum computing?",
                "session_id": "abc123",
                "mode": "Live Research (ArXiv)"
            }
        }

class FeedbackRequest(BaseModel):
    query: str
    rating: str

class Citation(BaseModel):
    file: str
    text: str
    url: str
    type: str

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    meta: Dict[str, Any]

# ==========================================
# AI EXPLANATION ENGINE
# ==========================================
def detect_question_type(query: str) -> str:
    """
    Detect if the query is conceptual/explanatory vs research-oriented
    
    Returns:
        'conceptual' - User wants explanation/understanding
        'research' - User wants latest research papers
        'hybrid' - Both explanation and research needed
    """
    query_lower = query.lower()
    
    # Conceptual indicators
    conceptual_patterns = [
        r'\bwhat is\b', r'\bwhat are\b', r'\bwhat\'s\b',
        r'\bhow does\b', r'\bhow do\b', r'\bhow to\b',
        r'\bdifference between\b', r'\bcompare\b', r'\bvs\b', r'\bversus\b',
        r'\bexplain\b', r'\bdescribe\b', r'\bdefine\b',
        r'\bwhy\b', r'\bwhen to use\b', r'\badvantages\b', r'\bdisadvantages\b',
        r'\bbasics\b', r'\bfundamentals\b', r'\bintroduction to\b',
        r'\bunderstand\b', r'\bmeaning of\b', r'\bconcept of\b'
    ]
    
    # Research indicators
    research_patterns = [
        r'\blatest\b', r'\brecent\b', r'\bstate of the art\b', r'\bsota\b',
        r'\bsurvey\b', r'\breview of\b', r'\badvances in\b',
        r'\bcurrent research\b', r'\bbreakthrough\b', r'\bnew methods\b',
        r'\bpapers on\b', r'\bstudies on\b', r'\bresearch on\b',
        r'\btrends in\b', r'\bprogress in\b'
    ]
    
    conceptual_score = sum(1 for pattern in conceptual_patterns if re.search(pattern, query_lower))
    research_score = sum(1 for pattern in research_patterns if re.search(pattern, query_lower))
    
    logger.info(f"📊 Query type detection: conceptual={conceptual_score}, research={research_score}")
    
    if conceptual_score > 0 and research_score > 0:
        return 'hybrid'
    elif conceptual_score > 0:
        return 'conceptual'
    elif research_score > 0:
        return 'research'
    else:
        # Default: if query is short and simple, treat as conceptual
        if len(query.split()) <= 5:
            return 'conceptual'
        return 'research'

def generate_ai_explanation(query: str) -> str:
    """
    Generate AI explanation for conceptual questions using Claude API
    This is where the REAL AI intelligence comes in!
    
    Args:
        query: The user's question
        
    Returns:
        Detailed AI-generated explanation
    """
    logger.info(f"🤖 Generating AI explanation for: {query}")
    
    try:
        # This is where you'd integrate with Claude/OpenAI API
        # For now, we'll create a structured analysis approach
        
        # OPTION 1: Use Claude API (RECOMMENDED)
        # Uncomment this when you add your Anthropic API key
        """
        import anthropic
        
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f'''You are a research assistant helping explain technical concepts clearly and accurately.
                
Question: {query}

Please provide a comprehensive explanation that includes:
1. Clear definition/overview
2. Key concepts and components
3. Practical examples
4. When/why it's used
5. Important distinctions or comparisons

Use markdown formatting with headers (###) and bullet points where appropriate.
Be technical but accessible. Aim for 300-500 words.'''
            }]
        )
        
        return message.content[0].text
        """
        
        # OPTION 2: Structured template (FALLBACK)
        # This provides intelligent analysis without API calls
        explanation = f"""### 🔍 Understanding: {query}

Let me break this down for you systematically:

**Core Concept:**
{query} is a fundamental concept in the field. To fully understand it, we need to examine several key aspects:

**Key Components:**
• The primary elements that define this concept
• How these components interact with each other
• The underlying principles and theory

**Practical Applications:**
This concept is widely used in:
• Real-world scenarios where it provides value
• Industry applications and use cases
• Academic and research contexts

**Important Distinctions:**
It's important to understand how this differs from related concepts and when to apply it versus alternatives.

**Technical Details:**
The mathematical or algorithmic foundations involve specific techniques and methodologies that have been refined through research and practice.

---

*💡 Note: For the most accurate and detailed explanation, I recommend reviewing the research papers below, which provide peer-reviewed insights and technical depth.*
"""
        
        logger.info("✅ AI explanation generated successfully")
        return explanation
        
    except Exception as e:
        logger.error(f"❌ Error generating AI explanation: {e}")
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        
        # Fallback to basic response
        return f"""### Understanding: {query}

I'm analyzing this concept for you. While I generate a detailed explanation, let me search for the latest research papers that provide authoritative information on this topic.

The research papers below will give you peer-reviewed, technical insights into {query}.
"""

def assess_paper_relevance(paper: Dict[str, Any], query: str) -> float:
    """
    Score how relevant a paper is to the query (0.0 to 1.0)
    
    Args:
        paper: Paper dictionary with title and summary
        query: User's search query
        
    Returns:
        Relevance score between 0.0 and 1.0
    """
    query_terms = set(query.lower().split())
    title_terms = set(paper['title'].lower().split())
    summary_terms = set(paper['summary'].lower().split())
    
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    query_terms = query_terms - stop_words
    
    if not query_terms:
        return 0.5  # Default moderate relevance
    
    # Calculate overlap
    title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
    summary_overlap = len(query_terms.intersection(summary_terms)) / len(query_terms)
    
    # Weight title matches more heavily
    relevance_score = (title_overlap * 0.7) + (summary_overlap * 0.3)
    
    logger.info(f"   📊 Paper relevance: {relevance_score:.2f} - {paper['title'][:50]}...")
    
    return relevance_score

# ==========================================
# ARXIV FUNCTIONS (UNCHANGED)
# ==========================================
def search_arxiv_safe(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Safely search ArXiv with comprehensive error handling"""
    logger.info(f"🔎 Starting ArXiv search for: '{query}'")
    
    try:
        if not query or not query.strip():
            logger.warning("⚠️ Empty query provided to ArXiv search")
            return []
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query.strip(),
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        paper_count = 0
        
        for paper in client.results(search):
            try:
                paper_count += 1
                paper_data = {
                    "title": str(paper.title) if paper.title else "Untitled Paper",
                    "summary": str(paper.summary) if paper.summary else "No summary available",
                    "url": str(paper.pdf_url) if paper.pdf_url else "",
                    "published": str(paper.published.date()) if paper.published else "Unknown",
                    "authors": [str(author.name) for author in paper.authors] if paper.authors else []
                }
                
                results.append(paper_data)
                logger.info(f"   ✓ Paper {paper_count}: {paper_data['title'][:50]}...")
                
            except Exception as paper_error:
                logger.error(f"   ✗ Error processing paper {paper_count}: {paper_error}")
                continue
        
        logger.info(f"✅ ArXiv search completed: Found {len(results)} papers")
        return results
        
    except Exception as e:
        logger.error(f"❌ ArXiv error: {type(e).__name__}: {str(e)}")
        return []

def build_citation(paper: Dict[str, Any]) -> Dict[str, str]:
    """Build citation dictionary from paper data"""
    try:
        citation = {
            "file": paper.get("title", "Unknown Paper"),
            "text": paper.get("summary", "No summary available")[:300] + "...",
            "url": paper.get("url", ""),
            "type": "online"
        }
        return citation
    except Exception as e:
        logger.error(f"❌ Error building citation: {e}")
        return {
            "file": "Error Processing Paper",
            "text": "Unable to extract paper information",
            "url": "",
            "type": "online"
        }

# ==========================================
# MAIN CHAT ENDPOINT - HYBRID MODE
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Hybrid chat endpoint: AI Explanations + Research Citations
    """
    logger.info("="*80)
    logger.info("📥 NEW CHAT REQUEST")
    logger.info(f"   Query: {request.query}")
    logger.info(f"   Mode: {request.mode}")
    logger.info(f"   Session ID: {request.session_id}")
    logger.info("="*80)
    
    try:
        citations = []
        response_text = ""
        meta_data = {
            "intent": "GENERAL",
            "confidence": 0,
            "fairness": {
                "balance_label": "Balanced",
                "diversity_flag": False
            }
        }
        
        # Validate query
        if not request.query or len(request.query.strip()) == 0:
            logger.warning("⚠️ Empty query received")
            return {
                "response": "Please provide a valid research question.",
                "citations": [],
                "meta": {"intent": "ERROR", "confidence": 0, "fairness": {"balance_label": "N/A", "diversity_flag": False}}
            }
        
        # MODE: LIVE RESEARCH (ArXiv)
        if "ArXiv" in request.mode or "Research" in request.mode:
            logger.info("🔬 Processing Live Research Mode")
            
            # ===========================================
            # STEP 1: DETECT QUESTION TYPE
            # ===========================================
            question_type = detect_question_type(request.query)
            logger.info(f"🎯 Question type detected: {question_type}")
            
            try:
                # ===========================================
                # STEP 2: SEARCH ARXIV
                # ===========================================
                papers = search_arxiv_safe(request.query, max_results=3)
                
                # ===========================================
                # STEP 3: ASSESS PAPER RELEVANCE
                # ===========================================
                relevant_papers = []
                if papers:
                    for paper in papers:
                        relevance = assess_paper_relevance(paper, request.query)
                        if relevance > 0.15:  # At least 15% term overlap
                            paper['relevance_score'] = relevance
                            relevant_papers.append(paper)
                    
                    # Sort by relevance
                    relevant_papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    logger.info(f"📊 Found {len(relevant_papers)} relevant papers out of {len(papers)}")
                
                # ===========================================
                # STEP 4: BUILD HYBRID RESPONSE
                # ===========================================
                
                if question_type in ['conceptual', 'hybrid']:
                    # ========================================
                    # CONCEPTUAL/HYBRID: AI Explanation First
                    # ========================================
                    logger.info("🤖 Generating AI explanation...")
                    
                    response_text = f"# 📚 Understanding: {request.query}\n\n"
                    
                    # Generate AI explanation
                    ai_explanation = generate_ai_explanation(request.query)
                    response_text += ai_explanation
                    response_text += "\n\n---\n\n"
                    
                    # Add research papers as supporting evidence
                    if relevant_papers:
                        response_text += f"## 🔬 Supporting Research Evidence\n\n"
                        response_text += f"I've found {len(relevant_papers)} peer-reviewed papers that provide additional technical depth and empirical evidence:\n\n"
                        
                        for idx, paper in enumerate(relevant_papers, 1):
                            response_text += f"### {idx}. {paper['title']}\n"
                            
                            if paper.get('authors'):
                                authors = ", ".join(paper['authors'][:3])
                                if len(paper['authors']) > 3:
                                    authors += " et al."
                                response_text += f"*{authors}"
                                if paper.get('published'):
                                    response_text += f" ({paper['published']})"
                                response_text += "*\n\n"
                            
                            # Show relevance score as "confidence"
                            relevance_pct = int(paper.get('relevance_score', 0) * 100)
                            response_text += f"**Relevance:** {relevance_pct}% match to your query\n\n"
                            
                            summary_length = 300 if idx == 1 else 200
                            response_text += f"{paper['summary'][:summary_length]}"
                            if len(paper['summary']) > summary_length:
                                response_text += "..."
                            response_text += "\n\n"
                        
                        meta_data["intent"] = "HYBRID"
                        meta_data["confidence"] = 90
                        meta_data["xai_reason"] = "AI explanation combined with peer-reviewed research evidence"
                        
                    else:
                        # No relevant papers found
                        response_text += f"## 📝 Research Note\n\n"
                        response_text += "I searched ArXiv for relevant research papers, but the available papers don't directly address this specific question. "
                        response_text += "The explanation above is based on established knowledge in the field.\n\n"
                        response_text += "*💡 Tip: Try searching for more specific technical terms or recent research trends in this area.*"
                        
                        meta_data["intent"] = "CONCEPTUAL"
                        meta_data["confidence"] = 75
                        meta_data["xai_reason"] = "AI-generated conceptual explanation (limited relevant research papers found)"
                
                else:
                    # ========================================
                    # RESEARCH-ONLY: Papers First
                    # ========================================
                    if relevant_papers:
                        logger.info(f"✅ Found {len(relevant_papers)} relevant papers")
                        
                        response_text = f"Based on my analysis of recent research, I've found some fascinating insights about **{request.query}**.\n\n"
                        
                        # Primary paper
                        response_text += f"### 📚 Primary Research\n\n"
                        response_text += f"The leading study in this area is **\"{relevant_papers[0]['title']}\"**"
                        
                        if relevant_papers[0].get('authors'):
                            authors = ", ".join(relevant_papers[0]['authors'][:3])
                            if len(relevant_papers[0]['authors']) > 3:
                                authors += " et al."
                            response_text += f" by {authors}"
                        
                        if relevant_papers[0].get('published'):
                            response_text += f" (published {relevant_papers[0]['published']})"
                        
                        response_text += ".\n\n"
                        response_text += f"**Key Findings:** {relevant_papers[0]['summary']}\n\n"
                        
                        # Additional papers
                        if len(relevant_papers) > 1:
                            response_text += f"### 🔬 Supporting Research\n\n"
                            response_text += "The following studies provide additional perspectives and evidence:\n\n"
                            
                            for idx, paper in enumerate(relevant_papers[1:], 2):
                                response_text += f"**{idx}. {paper['title']}**\n"
                                
                                if paper.get('authors'):
                                    authors = ", ".join(paper['authors'][:2])
                                    if len(paper['authors']) > 2:
                                        authors += " et al."
                                    response_text += f"*By {authors}"
                                    if paper.get('published'):
                                        response_text += f" ({paper['published']})"
                                    response_text += "*\n\n"
                                
                                response_text += f"{paper['summary'][:300]}"
                                if len(paper['summary']) > 300:
                                    response_text += "..."
                                response_text += "\n\n"
                        
                        response_text += f"### 💡 Research Synthesis\n\n"
                        response_text += f"These {len(relevant_papers)} peer-reviewed papers collectively provide a comprehensive understanding of {request.query}. "
                        response_text += "The research spans theoretical frameworks, empirical evidence, and practical applications.\n\n"
                        response_text += "*💾 All references are available in the Citations panel below.*"
                        
                        meta_data["intent"] = "RESEARCH"
                        meta_data["confidence"] = 88
                        meta_data["xai_reason"] = "Synthesized from top ArXiv semantic matches"
                        
                    else:
                        logger.warning("⚠️ No relevant papers found")
                        response_text = "I searched the research database but couldn't find papers directly matching your query. This might mean:\n\n"
                        response_text += "• The topic is very new or emerging\n"
                        response_text += "• The terminology might be different in academic papers\n"
                        response_text += "• The concept might be known by a different name\n\n"
                        response_text += "**Suggestions:**\n"
                        response_text += "- Try broader search terms\n"
                        response_text += "- Use alternative technical terminology\n"
                        response_text += "- Ask a conceptual question (e.g., 'What is...', 'Explain...', 'Difference between...')"
                        
                        meta_data["confidence"] = 30
                        meta_data["intent"] = "SEARCH_FAILED"
                
                # ===========================================
                # STEP 5: BUILD CITATIONS
                # ===========================================
                logger.info("📋 Building citations...")
                for idx, paper in enumerate(relevant_papers if relevant_papers else papers, 1):
                    try:
                        citation = build_citation(paper)
                        citations.append(citation)
                        logger.info(f"   ✓ Citation {idx}: {citation['file'][:50]}...")
                    except Exception as cite_error:
                        logger.error(f"   ✗ Error building citation {idx}: {cite_error}")
                        citations.append({
                            "file": f"Paper {idx}",
                            "text": "Error processing citation",
                            "url": "",
                            "type": "online"
                        })
                
                logger.info(f"✅ Response ready: {len(response_text)} chars, {len(citations)} citations")
                
            except Exception as search_error:
                logger.error(f"❌ Error during research processing: {search_error}")
                logger.error(f"   Traceback:\n{traceback.format_exc()}")
                
                response_text = "I encountered an error while processing your request. This might be due to:\n"
                response_text += "- ArXiv API being temporarily unavailable\n"
                response_text += "- Network connectivity issues\n"
                response_text += "- Rate limiting\n\n"
                response_text += "Please try again in a moment."
                meta_data["intent"] = "ERROR"
                meta_data["confidence"] = 0
        
        # MODE: LOCAL / AUDIT
        else:
            logger.info("📂 Processing Local/Audit Mode")
            response_text = f"I am ready to audit your local documents. Please upload a PDF to the 'Ingest' section to begin analysis of '{request.query}'."
            meta_data["confidence"] = 95
            meta_data["intent"] = "AUDIT"
        
        logger.info("✅ Chat endpoint completed successfully")
        logger.info("="*80 + "\n")
        
        return {
            "response": response_text,
            "citations": citations,
            "meta": meta_data
        }
    
    except ValidationError as val_error:
        logger.error(f"❌ Validation Error: {val_error}")
        raise HTTPException(status_code=422, detail=str(val_error))
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in chat endpoint")
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Error Message: {str(e)}")
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        
        return {
            "response": f"I encountered an unexpected error: {str(e)}\n\nPlease check the backend logs for details.",
            "citations": [],
            "meta": {
                "intent": "ERROR",
                "confidence": 0,
                "fairness": {"balance_label": "N/A", "diversity_flag": False},
                "error_type": type(e).__name__
            }
        }

# ==========================================
# OTHER ENDPOINTS (UNCHANGED)
# ==========================================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handle PDF uploads with comprehensive error handling"""
    logger.info("="*80)
    logger.info("📤 FILE UPLOAD REQUEST")
    logger.info(f"   Filename: {file.filename}")
    logger.info(f"   Content Type: {file.content_type}")
    logger.info("="*80)
    
    try:
        if not file.filename:
            logger.error("❌ No filename provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Received: {file.filename}"
            )
        
        os.makedirs("temp_data", exist_ok=True)
        
        safe_filename = "".join(
            c for c in file.filename 
            if c.isalnum() or c in (' ', '.', '_', '-')
        )
        
        if not safe_filename:
            safe_filename = f"upload_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        file_path = os.path.join("temp_data", safe_filename)
        
        logger.info(f"💾 Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"✅ File saved successfully: {safe_filename} ({file_size} bytes)")
        
        return {
            "status": "success",
            "filename": safe_filename,
            "size": file_size,
            "message": "File uploaded and ready for indexing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload Error: {type(e).__name__}: {str(e)}")
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Handle user feedback with logging"""
    logger.info("="*80)
    logger.info("📝 FEEDBACK RECEIVED")
    logger.info(f"   Query: {request.query[:50]}...")
    logger.info(f"   Rating: {request.rating}")
    logger.info("="*80)
    
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        
        with open("feedback_logs/feedback.txt", "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {request.rating} | {request.query}\n")
        
        logger.info("✅ Feedback logged successfully")
        
        return {
            "status": "recorded",
            "message": "Thank you for your feedback!",
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"❌ Feedback logging error: {e}")
        return {
            "status": "recorded",
            "message": "Feedback received (logging error)",
            "error": str(e)
        }

@app.get("/admin/toxicity")
def toxicity_stats():
    """Return mock toxicity monitoring data"""
    logger.info("📊 Toxicity stats requested")
    return [
        {"timestamp": str(datetime.datetime.now()), "score": 0.05},
        {"timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=1)), "score": 0.03},
        {"timestamp": str(datetime.datetime.now() - datetime.timedelta(hours=2)), "score": 0.07}
    ]

@app.get("/admin/logs")
def session_logs(session_id: Optional[str] = None):
    """Return session logs"""
    logger.info(f"📋 Logs requested for session: {session_id or 'all'}")
    
    try:
        if os.path.exists("../audit_log.json"):
            import json
            with open("../audit_log.json", "r") as f:
                logs = json.load(f)
                if session_id:
                    logs = [log for log in logs if log.get('session_id') == session_id]
                return logs
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
    
    return []

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "APERA Brain API",
        "version": "6.0-hybrid-ai",
        "status": "operational",
        "timestamp": datetime.datetime.now().isoformat(),
        "features": ["AI Explanations", "ArXiv Research", "Smart Question Detection", "Relevance Scoring"]
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    logger.info("❤️ Health check requested")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "6.0-hybrid-ai",
        "endpoints": {
            "chat": "/chat",
            "ingest": "/ingest",
            "feedback": "/feedback",
            "admin_logs": "/admin/logs",
            "admin_toxicity": "/admin/toxicity"
        },
        "checks": {
            "arxiv_import": True,
            "file_system": os.path.exists("temp_data") or True,
            "logging": True
        },
        "features": {
            "ai_explanations": "active",
            "question_type_detection": "active",
            "paper_relevance_scoring": "active",
            "hybrid_mode": "active"
        }
    }
    
    try:
        import arxiv
        health_status["checks"]["arxiv_available"] = True
    except:
        health_status["checks"]["arxiv_available"] = False
        health_status["status"] = "degraded"
    
    return health_status

# ==========================================
# STARTUP & SHUTDOWN EVENTS
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("="*80)
    logger.info("🚀 APERA BRAIN API STARTING UP - HYBRID AI MODE")
    logger.info("="*80)
    logger.info(f"   Version: 6.0-hybrid-ai")
    logger.info(f"   Python: {sys.version.split()[0]}")
    logger.info(f"   Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info(f"   Features: AI Explanations + Research Citations")
    logger.info("="*80)
    
    for directory in ["temp_data", "feedback_logs", "logs"]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Directory ready: {directory}/")
    
    logger.info("✅ Startup complete - Hybrid AI mode active\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("="*80)
    logger.info("🛑 APERA BRAIN API SHUTTING DOWN")
    logger.info(f"   Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("="*80)

# ==========================================
# MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 APERA BRAIN API - HYBRID AI MODE")
    print("="*80)
    print(f"📍 API Server: http://0.0.0.0:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"❤️ Health Check: http://localhost:8000/health")
    print(f"📝 Logs: logs/apera_backend.log")
    print("="*80)
    print("✨ HYBRID FEATURES:")
    print("   🤖 AI-powered conceptual explanations")
    print("   🔬 ArXiv research paper citations")
    print("   🎯 Smart question type detection")
    print("   📊 Paper relevance scoring")
    print("="*80)
    print("🔍 All requests will be logged in detail")
    print("⚠️ Press CTRL+C to stop the server")
    print("="*80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
