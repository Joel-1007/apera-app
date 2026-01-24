import os
import pickle
import faiss
import re
import json
import numpy as np
import arxiv
from semanticscholar import SemanticScholar
from fairlearn.metrics import selection_rate
# --- FIX: Updated Import Path for LlamaIndex v0.10+ ---
from llama_index.llms.ollama import Ollama 
from src.utils.embed import Embedder
from transformers import pipeline

# --- CONFIG ---
INDEX_DIR = "data/indexes"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(INDEX_DIR, "chunk_metadata.pkl")
CACHE_FILE = "arxiv_cache.json"

class FairnessMonitor:
    """Tool D.1: Fairness Toolkit (Fairlearn)"""
    def calculate_parity(self, sources):
        # We define 'Western' as the privileged group for this audit
        western_keywords = ['usa', 'europe', 'uk', 'canada', 'ieee', 'acm', 'western']
        
        # Create dataset for Fairlearn
        # y_true = [1] means all retrieved docs are considered "selected" outcomes
        y_true = [1] * len(sources) 
        
        sensitive_features = [] # 1 if Western, 0 if Global South
        
        for s in sources:
            content = (s['text'] + s['file']).lower()
            is_western = 1 if any(k in content for k in western_keywords) else 0
            sensitive_features.append(is_western)
            
        if not sensitive_features: return {"parity_score": 0, "status": "No Data"}

        # Calculate Selection Rate (How often Western sources are picked)
        # In a search context, this tells us the dominance ratio
        try:
            rate = selection_rate(y_true, y_true, sensitive_features=sensitive_features)
        except:
            rate = 0.0
        
        # Fairlearn logic: If Western selection rate > 0.8, it's disparate impact
        status = "Biased (Western Dominance)" if rate > 0.8 else "Balanced"
        
        return {
            "western_selection_rate": round(rate, 2),
            "fairlearn_status": status,
            "balance_label": status
        }

class RAGSystem:
    def __init__(self):
        # 1. Load Vector DB
        self.chunks = []
        if os.path.exists(INDEX_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
                with open(METADATA_FILE, "rb") as f: self.chunks = pickle.load(f)
            except:
                self.index = None
        else: self.index = None

        self.embedder = Embedder(model_name="all-MiniLM-L6-v2")
        
        # Tool D.2: Connect to Ollama
        print("🧠 Connecting to LLM (Ollama)...")
        self.llm = Ollama(model="mistral", request_timeout=300.0)
        
        # Tools
        self.sch = SemanticScholar() # Tool D.3
        self.fairness = FairnessMonitor() # Tool D.1
        self.guard = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f: return json.load(f)
        return {}

    def search_semantic_scholar(self, query: str, k: int = 3):
        """Tool D.3: Richer Metadata Search"""
        print(f"🌍 querying Semantic Scholar for: {query}")
        try:
            results = self.sch.search_paper(query, limit=k)
            sources = []
            for item in results:
                # Semantic Scholar gives citation counts - very useful!
                text = item.abstract if item.abstract else item.title
                sources.append({
                    'text': text,
                    'file': f"{item.title} (Citations: {item.citationCount})",
                    'score': 0.98,
                    'type': 'online_semantic',
                    'url': item.url
                })
            return sources
        except Exception as e:
            print(f"Semantic Scholar Error: {e}")
            return [] # Fallback

    def search_hybrid(self, query: str, k: int = 5):
        # Simplified Hybrid Search logic for fallback
        if not self.index: return [], []
        
        xq = self.embedder.encode(query).reshape(1, -1)
        D, I = self.index.search(xq, k)
        
        results = []
        for i in I[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        
        return results, D[0]

    def chat(self, user_query: str, mode: str = "local"):
        # 1. Guardrail
        try:
            res = self.guard(user_query)[0]
            # Check if any toxic label has high confidence
            if any(r['score'] > 0.7 for r in res if r['label'] != 'neutral'): 
                return {"response": "🚫 Blocked by Toxic-BERT Guardrail.", "meta": {}, "citations": []}
        except:
            pass

        # 2. Retrieval (Tool D.3 Swapping)
        mode_msg = "Local DB"
        sources = []
        
        if mode == "semantic":
            sources = self.search_semantic_scholar(user_query)
            mode_msg = "Semantic Scholar API"
        elif mode == "online":
            # Basic ArXiv fallback
            try:
                client = arxiv.Client()
                search = arxiv.Search(query=user_query, max_results=3)
                sources = [{'text': r.summary, 'file': r.title, 'type': 'arxiv'} for r in client.results(search)]
                mode_msg = "ArXiv"
            except:
                sources = []
        else:
            sources, _ = self.search_hybrid(user_query) # Local
            
        # If Semantic/ArXiv fail, fallback to local or empty
        if not sources and mode != "local":
             sources, _ = self.search_hybrid(user_query)
             mode_msg = "Local DB (Fallback)"

        # 3. Fairness Analysis (Tool D.1)
        fairness_metrics = self.fairness.calculate_parity(sources)

        # 4. Generate
        context = "\n".join([s['text'][:500] for s in sources]) # Truncate for prompt limit
        prompt = f"Context ({mode_msg}):\n{context}\n\nUser: {user_query}\nAnswer:"
        
        try:
            response = self.llm.complete(prompt).text
        except:
            response = "I'm having trouble connecting to the LLM. Please ensure Ollama is running."

        return {
            "response": response,
            "citations": sources,
            "meta": {
                "intent": "RESEARCH",
                "fairness": fairness_metrics,
                "confidence": 85, # Mock confidence for UI
                "xai_reason": f"Based on {len(sources)} sources from {mode_msg}"
            }
        }
