# APERA Pro: Advanced Production-Grade Research Intelligence
**A Hybrid, Ethical, and Live-Updating AI Research Assistant**

![Status](https://img.shields.io/badge/Status-Live_Beta-blueviolet)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid_Cloud-blue)
![Focus](https://img.shields.io/badge/Focus-Ethical_AI-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

## 📖 Executive Summary
**APERA Pro** (Advanced Production-Grade Ethical Research Assistant) is a proof-of-concept AI system designed to solve the "Knowledge Cutoff" problem in academic research.

Unlike standard chatbots that rely on frozen training data, APERA Pro uses a **Hybrid RAG (Retrieval-Augmented Generation)** architecture. It connects a cloud-based user interface to a local research engine that fetches, analyzes, and synthesizes **live pre-print papers from arXiv (2024- Beyond)** in real-time.

The system emphasizes **AI Ethics** with a built-in Governance Dashboard that monitors toxicity, hallucination risks, and citation bias (Global North vs. Global South representation).

---

## 🏗️ System Architecture
APERA Pro implements a **"Split-Brain" Strategy** to balance performance, cost, and security.

graph TD
    User[👤 User via Browser] <-->|HTTPS| Streamlit[☁️ Streamlit Cloud UI]
    Streamlit <-->|Encrypted Tunnel| Ngrok[🔒 Ngrok Gateway]
    Ngrok <-->|REST API| FastAPI[⚡ Local FastAPI Server]
    
    subgraph "🧠 Local Secure Environment (The Brain)"
        FastAPI --> Agent[🤖 RAG Agent / Orchestrator]
        Agent --> Arxiv[📚 Live arXiv API]
        Agent --> VectorDB[(🗄️ FAISS Vector DB)]
        Agent --> LLM[🦙 Ollama / Llama3]
        Agent --> Hybrid[🔍 Hybrid Search Engine]
        
        VectorDB <--> Embed[🔢 SentenceTransformers]
        Hybrid --> VectorDB
        Hybrid --> BM25[📝 BM25 Keyword Search]
        
        Agent --> Governance[🛡️ Ethics & Audit Module]
        Governance --> Toxicity[☠️ Toxicity Detector]
        Governance --> Bias[🌍 Geographic Bias Analyzer]
        Governance --> Hallucination[⚠️ Confidence Scorer]
    end
    
    FastAPI --> Database[(💾 SQLite Audit DB)]
    
    style User fill:#e1f5ff
    style Streamlit fill:#ff6b6b
    style Ngrok fill:#ffd93d
    style FastAPI fill:#6bcf7f
    style Agent fill:#a29bfe
    style LLM fill:#fd79a8
    style VectorDB fill:#fdcb6e
    style Governance fill:#00b894


    
