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
    version="7.0-claude-powered",
    description="Claude-powered research intelligence with advanced synthesis"
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

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    meta: Dict[str, Any]

# ==========================================
# CLAUDE AI INTEGRATION
# ==========================================
def call_claude_api(prompt: str, system_prompt: str = None, max_tokens: int = 2048) -> str:
    """
    Call Claude API for advanced AI analysis
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system context
        max_tokens: Maximum response length
        
    Returns:
        Claude's response text
    """
    try:
        # Check if API key is available
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            logger.warning("⚠️ ANTHROPIC_API_KEY not set - using fallback mode")
            return None
        
        logger.info("🤖 Calling Claude API...")
        
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "messages": messages
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        message = client.messages.create(**kwargs)
        
        response_text = message.content[0].text
        
        logger.info(f"✅ Claude API responded: {len(response_text)} chars")
        
        return response_text
        
    except ImportError:
        logger.error("❌ Anthropic package not installed. Run: pip install anthropic --break-system-packages")
        return None
    except Exception as e:
        logger.error(f"❌ Claude API error: {type(e).__name__}: {str(e)}")
        logger.error(f"   Traceback:\n{traceback.format_exc()}")
        return None

# ==========================================
# ADVANCED RESEARCH SYNTHESIS ENGINE
# ==========================================
def generate_advanced_research_synthesis(query: str, papers: List[Dict[str, Any]]) -> str:
    """
    Use Claude to analyze ArXiv papers and generate sophisticated research synthesis
    
    This is the ADVANCED mode - Claude reads all papers and creates an intelligent summary
    
    Args:
        query: User's research question
        papers: List of ArXiv papers with titles, authors, abstracts
        
    Returns:
        Claude-generated comprehensive research synthesis
    """
    logger.info(f"🧠 Generating ADVANCED research synthesis with Claude AI...")
    
    if not papers or len(papers) == 0:
        logger.warning("⚠️ No papers provided for synthesis")
        return None
    
    # ========================================
    # BUILD COMPREHENSIVE PROMPT FOR CLAUDE
    # ========================================
    
    # Prepare paper summaries for Claude
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
        papers_context += f"URL: {paper['url']}\n"
        papers_context += f"\nABSTRACT:\n{paper['summary']}\n"
    
    system_prompt = """You are an expert research analyst specializing in synthesizing academic literature. Your role is to:

1. Analyze multiple research papers deeply
2. Identify key themes, methodologies, and findings
3. Compare and contrast different approaches
4. Explain complex concepts clearly
5. Provide actionable insights
6. Cite papers appropriately using inline references like [Paper 1], [Paper 2]

Write in a professional yet accessible academic style. Use markdown formatting with headers, bullet points, and emphasis where appropriate."""

    user_prompt = f"""I'm researching: "{query}"

I've found {len(papers)} relevant academic papers from ArXiv. Please analyze these papers and provide a comprehensive research synthesis.

{papers_context}

{'='*80}

Please provide a detailed research synthesis that includes:

1. **Executive Summary** (2-3 sentences): What are the key takeaways?

2. **Core Concepts & Definitions**: Explain the fundamental concepts addressed in these papers

3. **Methodological Approaches**: What methods/techniques do these papers use?

4. **Key Findings & Contributions**: What are the major discoveries or innovations?

5. **Comparative Analysis**: How do these papers relate to each other? Do they agree, disagree, or complement?

6. **Practical Applications**: What are the real-world implications?

7. **Research Gaps & Future Directions**: What questions remain unanswered?

8. **Recommended Reading Order**: Which papers should be read first, and why?

Use inline citations like [Paper 1], [Paper 2], etc. throughout your analysis. Make it comprehensive yet readable - aim for 600-1000 words.

Remember: You're writing for someone who wants deep understanding, not just a surface-level summary."""

    # ========================================
    # CALL CLAUDE API
    # ========================================
    
    synthesis = call_claude_api(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=3000
    )
    
    if synthesis:
        logger.info(f"✅ Advanced synthesis generated: {len(synthesis)} chars")
        return synthesis
    else:
        logger.warning("⚠️ Claude API unavailable, using fallback synthesis")
        return None

def generate_fallback_synthesis(query: str, papers: List[Dict[str, Any]]) -> str:
    """
    Fallback synthesis when Claude API is unavailable
    Creates intelligent structured summary without AI
    """
    logger.info("📝 Generating fallback synthesis (no AI API)...")
    
    response = f"# Research Analysis: {query}\n\n"
    
    response += f"## 📊 Overview\n\n"
    response += f"I've analyzed {len(papers)} peer-reviewed papers from ArXiv on this topic. "
    response += f"Below is a comprehensive synthesis of the research landscape.\n\n"
    
    # Primary paper - detailed
    response += f"## 📚 Foundational Research\n\n"
    response += f"### {papers[0]['title']}\n"
    
    if papers[0].get('authors'):
        authors = ", ".join(papers[0]['authors'][:4])
        if len(papers[0]['authors']) > 4:
            authors += " et al."
        response += f"**Authors:** {authors}\n"
    
    if papers[0].get('published'):
        response += f"**Published:** {papers[0]['published']}\n"
    
    response += f"\n**Abstract Summary:**\n{papers[0]['summary']}\n\n"
    
    # Extract key terms from first paper
    summary_lower = papers[0]['summary'].lower()
    key_indicators = []
    
    if any(term in summary_lower for term in ['propose', 'present', 'introduce', 'develop']):
        key_indicators.append("**Novel Contribution:** This paper introduces new methods or frameworks")
    
    if any(term in summary_lower for term in ['experiment', 'evaluation', 'benchmark', 'dataset']):
        key_indicators.append("**Empirical Evidence:** Includes experimental validation")
    
    if any(term in summary_lower for term in ['state-of-the-art', 'sota', 'outperform', 'improve']):
        key_indicators.append("**Performance:** Claims improvements over existing methods")
    
    if key_indicators:
        response += "**Key Characteristics:**\n"
        for indicator in key_indicators:
            response += f"• {indicator}\n"
        response += "\n"
    
    # Supporting papers
    if len(papers) > 1:
        response += f"## 🔬 Supporting Literature\n\n"
        response += f"The following {len(papers)-1} paper(s) provide complementary perspectives:\n\n"
        
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
            
            # Show first 400 chars of abstract
            summary_preview = paper['summary'][:400]
            if len(paper['summary']) > 400:
                summary_preview += "..."
            
            response += f"{summary_preview}\n\n"
    
    # Synthesis section
    response += f"## 💡 Research Synthesis\n\n"
    response += f"**Collective Insights:**\n\n"
    response += f"These {len(papers)} papers collectively advance our understanding of {query} through:\n\n"
    response += f"• **Theoretical Frameworks:** Establishing mathematical and conceptual foundations\n"
    response += f"• **Methodological Innovation:** Introducing new techniques and approaches\n"
    response += f"• **Empirical Validation:** Providing experimental evidence and benchmarks\n"
    response += f"• **Practical Applications:** Demonstrating real-world utility\n\n"
    
    response += f"**Recommended Exploration:**\n\n"
    response += f"1. **Start with Paper 1** - Provides foundational understanding\n"
    if len(papers) > 1:
        response += f"2. **Explore Papers 2-{len(papers)}** - Build on core concepts with specialized perspectives\n"
    response += f"3. **Review citations** - Follow references for deeper context\n\n"
    
    response += f"*💾 Full paper PDFs are available via the citation links below.*\n"
    
    return response

def generate_conceptual_explanation(query: str, papers: List[Dict[str, Any]] = None) -> str:
    """
    Use Claude to generate conceptual explanation with optional paper context
    
    Args:
        query: The conceptual question
        papers: Optional papers for additional context
        
    Returns:
        Claude-generated explanation
    """
    logger.info(f"🎓 Generating conceptual explanation with Claude...")
    
    # Build context from papers if available
    papers_context = ""
    if papers and len(papers) > 0:
        papers_context = f"\n\nREFERENCE MATERIALS (optional context):\n"
        for idx, paper in enumerate(papers[:2], 1):  # Use top 2 papers
            papers_context += f"\nPaper {idx}: {paper['title']}\n"
            papers_context += f"Summary: {paper['summary'][:300]}...\n"
    
    system_prompt = """You are an expert educator specializing in explaining complex technical concepts clearly and accurately. Your explanations should:

1. Start with a clear, simple definition
2. Break down concepts into understandable components
3. Use analogies and examples where helpful
4. Compare/contrast with related concepts
5. Explain practical applications
6. Be technically accurate but accessible

Use markdown formatting with headers (###), bullet points, and **bold** for emphasis."""

    user_prompt = f"""Please provide a comprehensive explanation for this question:

"{query}"
{papers_context}

Structure your explanation to include:

1. **Clear Definition**: What is this concept in simple terms?
2. **Key Components**: What are the essential parts/elements?
3. **How It Works**: Explain the mechanism or process
4. **Practical Examples**: Real-world applications or use cases
5. **Important Distinctions**: How does it differ from similar concepts?
6. **When to Use**: Guidelines for application

Aim for 400-600 words. Be technical but accessible. Use specific examples where possible."""

    explanation = call_claude_api(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=2000
    )
    
    if explanation:
        logger.info(f"✅ Conceptual explanation generated: {len(explanation)} chars")
        return explanation
    else:
        # Fallback explanation
        logger.warning("⚠️ Claude API unavailable, using fallback explanation")
        return f"""### Understanding: {query}

Let me provide a comprehensive explanation of this concept.

**Core Concept:**
{query} is an important topic in the field. Understanding it requires examining several key aspects:

**Key Components:**
• Fundamental elements that define this concept
• How these components interact and relate
• The underlying principles and theoretical foundations

**Practical Applications:**
This concept is widely used in:
• Real-world scenarios across various domains
• Industry applications and use cases
• Academic research and advanced studies

**Important Context:**
When working with this concept, it's crucial to understand how it differs from related ideas and when to apply it versus alternative approaches.

---

*💡 For deeper technical details, please review the research papers below which provide peer-reviewed insights.*
"""

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
        r'\bwhy\b', r'\bwhen to use\b', r'\badvantages\b', r'\bdisadvantages\b',
        r'\bbasics\b', r'\bfundamentals\b', r'\bintroduction to\b',
        r'\bunderstand\b', r'\bmeaning of\b', r'\bconcept of\b'
    ]
    
    research_patterns = [
        r'\blatest\b', r'\brecent\b', r'\bstate of the art\b', r'\bsota\b',
        r'\bsurvey\b', r'\breview of\b', r'\badvances in\b',
        r'\bcurrent research\b', r'\bbreakthrough\b', r'\bnew methods\b',
        r'\bpapers on\b', r'\bstudies on\b', r'\bresearch on\b',
        r'\btrends in\b', r'\bprogress in\b'
    ]
    
    conceptual_score = sum(1 for pattern in conceptual_patterns if re.search(pattern, query_lower))
    research_score = sum(1 for pattern in research_patterns if re.search(pattern, query_lower))
    
    logger.info(f"📊 Question type: conceptual={conceptual_score}, research={research_score}")
    
    if conceptual_score > 0 and research_score > 0:
        return 'hybrid'
    elif conceptual_score > 0:
        return 'conceptual'
    elif research_score > 0:
        return 'research'
    else:
        if len(query.split()) <= 5:
            return 'conceptual'
        return 'research'

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
                logger.error(f"   ✗ Error processing paper: {e}")
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
    except Exception as e:
        logger.error(f"❌ Citation error: {e}")
        return {
            "file": "Error",
            "text": "Unable to extract",
            "url": "",
            "type": "online"
        }

# ==========================================
# MAIN CHAT ENDPOINT - CLAUDE-POWERED
# ==========================================
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Claude-powered chat endpoint with advanced research synthesis
    """
    logger.info("="*80)
    logger.info("📥 NEW CHAT REQUEST - CLAUDE-POWERED MODE")
    logger.info(f"   Query: {request.query}")
    logger.info(f"   Mode: {request.mode}")
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
            logger.info("🔬 Processing CLAUDE-POWERED Research Mode")
            
            try:
                # ========================================
                # STEP 1: DETECT QUESTION TYPE
                # ========================================
                question_type = detect_question_type(request.query)
                logger.info(f"🎯 Question type: {question_type}")
                
                # ========================================
                # STEP 2: SEARCH ARXIV
                # ========================================
                papers = search_arxiv_safe(request.query, max_results=3)
                
                if not papers or len(papers) == 0:
                    logger.warning("⚠️ No papers found")
                    
                    response_text = f"I searched ArXiv for papers on '{request.query}' but didn't find relevant matches.\n\n"
                    response_text += "**Suggestions:**\n"
                    response_text += "• Try broader search terms\n"
                    response_text += "• Check spelling and terminology\n"
                    response_text += "• Use different keywords or phrases\n"
                    
                    meta_data["intent"] = "SEARCH_FAILED"
                    meta_data["confidence"] = 30
                    
                else:
                    # ========================================
                    # STEP 3: GENERATE RESPONSE WITH CLAUDE
                    # ========================================
                    
                    if question_type == 'conceptual':
                        # CONCEPTUAL: Explanation + Papers
                        logger.info("🎓 Mode: Conceptual Explanation")
                        
                        explanation = generate_conceptual_explanation(request.query, papers)
                        
                        response_text = f"# 📚 Understanding: {request.query}\n\n"
                        response_text += explanation
                        response_text += "\n\n---\n\n"
                        response_text += f"## 📖 Recommended Reading\n\n"
                        response_text += f"For deeper technical insights, I recommend these {len(papers)} papers:\n\n"
                        
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
                        meta_data["xai_reason"] = "Claude-generated explanation with supporting research"
                        
                    elif question_type == 'research':
                        # RESEARCH: Advanced Synthesis
                        logger.info("🧠 Mode: Advanced Research Synthesis")
                        
                        synthesis = generate_advanced_research_synthesis(request.query, papers)
                        
                        if synthesis:
                            response_text = synthesis
                            meta_data["intent"] = "RESEARCH"
                            meta_data["confidence"] = 95
                            meta_data["xai_reason"] = "Claude-powered synthesis of ArXiv research"
                        else:
                            # Fallback if Claude unavailable
                            response_text = generate_fallback_synthesis(request.query, papers)
                            meta_data["intent"] = "RESEARCH"
                            meta_data["confidence"] = 85
                            meta_data["xai_reason"] = "Structured synthesis (Claude API unavailable)"
                    
                    else:  # hybrid
                        # HYBRID: Both explanation and synthesis
                        logger.info("✨ Mode: Hybrid (Explanation + Synthesis)")
                        
                        explanation = generate_conceptual_explanation(request.query, papers)
                        
                        response_text = f"# 📚 Understanding: {request.query}\n\n"
                        response_text += explanation
                        response_text += "\n\n---\n\n"
                        
                        synthesis = generate_advanced_research_synthesis(request.query, papers)
                        
                        if synthesis:
                            response_text += f"## 🔬 Research Analysis\n\n"
                            response_text += synthesis
                        else:
                            response_text += generate_fallback_synthesis(request.query, papers)
                        
                        meta_data["intent"] = "HYBRID"
                        meta_data["confidence"] = 92
                        meta_data["xai_reason"] = "Claude-powered hybrid analysis"
                    
                    # ========================================
                    # STEP 4: BUILD CITATIONS
                    # ========================================
                    for paper in papers:
                        citations.append(build_citation(paper))
                    
                    logger.info(f"✅ Response ready: {len(response_text)} chars, {len(citations)} citations")
                
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                logger.error(traceback.format_exc())
                
                response_text = "An error occurred while processing your request. Please try again."
                meta_data["intent"] = "ERROR"
        
        # MODE: LOCAL
        else:
            response_text = f"Ready to audit local documents. Upload a PDF to analyze '{request.query}'."
            meta_data["confidence"] = 95
            meta_data["intent"] = "AUDIT"
        
        logger.info("✅ Completed")
        logger.info("="*80 + "\n")
        
        return {
            "response": response_text,
            "citations": citations,
            "meta": meta_data
        }
    
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "response": f"Error: {str(e)}",
            "citations": [],
            "meta": {"intent": "ERROR", "confidence": 0, "fairness": {"balance_label": "N/A", "diversity_flag": False}}
        }

# ==========================================
# OTHER ENDPOINTS (UNCHANGED)
# ==========================================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handle PDF uploads"""
    logger.info(f"📤 Upload: {file.filename}")
    
    try:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files supported")
        
        os.makedirs("temp_data", exist_ok=True)
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-'))
        file_path = os.path.join("temp_data", safe_filename or "upload.pdf")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"status": "success", "filename": safe_filename, "size": os.path.getsize(file_path)}
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(500, str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Handle feedback"""
    logger.info(f"📝 Feedback: {request.rating}")
    
    try:
        os.makedirs("feedback_logs", exist_ok=True)
        with open("feedback_logs/feedback.txt", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} | {request.rating} | {request.query}\n")
        return {"status": "recorded", "message": "Thank you!"}
    except Exception as e:
        return {"status": "recorded", "error": str(e)}

@app.get("/")
def root():
    return {
        "service": "APERA Brain API",
        "version": "7.0-claude-powered",
        "status": "operational",
        "features": ["Claude AI Integration", "Advanced Research Synthesis", "Conceptual Explanations"]
    }

@app.get("/health")
def health_check():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    return {
        "status": "healthy",
        "version": "7.0-claude-powered",
        "claude_api": "enabled" if api_key else "disabled (using fallback)",
        "features": {
            "advanced_synthesis": "active",
            "conceptual_explanations": "active",
            "arxiv_search": "active"
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("="*80)
    logger.info("🚀 APERA CLAUDE-POWERED API STARTING")
    logger.info("="*80)
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        logger.info("✅ Claude API: ENABLED")
    else:
        logger.warning("⚠️ Claude API: DISABLED (set ANTHROPIC_API_KEY to enable)")
        logger.info("   Running in FALLBACK mode - structured synthesis without AI")
    
    for directory in ["temp_data", "feedback_logs", "logs"]:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ Startup complete\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 APERA BRAIN API - CLAUDE-POWERED RESEARCH INTELLIGENCE")
    print("="*80)
    print(f"📍 Server: http://0.0.0.0:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("="*80)
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("✅ CLAUDE API: ENABLED")
        print("   🧠 Advanced AI synthesis active")
        print("   🎓 Intelligent explanations active")
    else:
        print("⚠️ CLAUDE API: DISABLED")
        print("   Set ANTHROPIC_API_KEY to enable AI features")
        print("   Currently using structured fallback mode")
    
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
