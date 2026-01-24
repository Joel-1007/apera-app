import os
import pickle
import faiss
import numpy as np

# Imports for LlamaIndex 0.9.x
from llama_index.agent import ReActAgent
from llama_index.tools import FunctionTool
from llama_index.llms import Ollama
from src.utils.embed import Embedder

# Configuration
INDEX_DIR = "data/indexes"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(INDEX_DIR, "chunk_metadata.pkl")

class KnowledgeBase:
    def __init__(self):
        print("⚙️ Loading Knowledge Base...")
        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(f"Missing index! Run rag_core.py first.")
        self.index = faiss.read_index(INDEX_FILE)
        
        with open(METADATA_FILE, "rb") as f:
            self.chunks = pickle.load(f)
            
        self.embedder = Embedder(model_name="all-MiniLM-L6-v2")
        print(f"✅ Loaded {len(self.chunks)} documents.")

    def search(self, query: str) -> str:
        print(f"\n🔎 Searching Database for: '{query}'...")
        query_vec = self.embedder.get_embeddings([query])
        D, I = self.index.search(query_vec, 3)
        
        results = []
        for idx in I[0]:
            if idx == -1: continue
            chunk = self.chunks[idx]
            info = (
                f"Source: {os.path.basename(chunk['source'])}\n"
                f"Content: {chunk['text']}\n"
            )
            results.append(info)
            
        return "\n---\n".join(results)

def setup_agent():
    print("🚀 Connecting to local Ollama (Gemma 3:4b)...")
    
    # UPDATED: Using Gemma 3:4b as seen in your screenshot
    llm = Ollama(model="gemma2:2b", request_timeout=300.0)

    kb = KnowledgeBase()
    search_tool = FunctionTool.from_defaults(
        fn=kb.search,
        name="search_academic_papers",
        description="Search for details in AI ethics research papers."
    )

    agent = ReActAgent.from_tools(
        [search_tool], 
        llm=llm, 
        verbose=True,
        max_iterations=5
    )
    
    return agent

if __name__ == "__main__":
    try:
        agent = setup_agent()
        print("\n🤖 APERA Agent Ready (Local Mode)! (Type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Make sure the Ollama app is running!")
