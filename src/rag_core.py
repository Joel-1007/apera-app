import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from src.utils.chunking import PDFProcessor
from src.utils.embed import Embedder

# Configuration
DATA_DIR = "data/raw"
INDEX_DIR = "data/indexes"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(INDEX_DIR, "chunk_metadata.pkl")

def build_pipeline():
    # 1. Initialize Components
    print("⚙️ Initializing RAG components...")
    processor = PDFProcessor(chunk_size=512, chunk_overlap=128)
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    
    # 2. Load PDFs
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDFs found in {DATA_DIR}. Run data_fetch.py first.")
        return

    print(f"📚 Found {len(pdf_files)} PDFs to process.")

    all_chunks = []
    all_embeddings = []

    # 3. Process Papers (Chunk & Embed)
    print("🚀 Starting ingestion pipeline...")
    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        filepath = os.path.join(DATA_DIR, filename)
        
        # A. Chunking
        chunks = processor.load_and_chunk(filepath)
        if not chunks:
            continue
            
        # B. Embedding
        texts = [c["text"] for c in chunks]
        embeddings = embedder.get_embeddings(texts)
        
        # C. Aggregate
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    # 4. Build FAISS Index
    if all_embeddings:
        # Concatenate all embeddings into one large matrix
        training_data = np.vstack(all_embeddings)
        dimension = training_data.shape[1]
        
        print(f"🧠 Building FAISS index for {len(all_chunks)} chunks (Dim: {dimension})...")
        
        # Create IVF Index (as per Guide Phase 3.2)
        # We use a simple FlatL2 index for <10k chunks, or IVF for larger scales.
        # For 100 PDFs, FlatL2 is faster and more accurate, but here is IVF setup for learning:
        quantizer = faiss.IndexFlatL2(dimension)
        # nlist = number of clusters (sqrt of N is a good rule of thumb)
        nlist = int(np.sqrt(len(all_chunks)))
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train and add
        index.train(training_data)
        index.add(training_data)
        
        # 5. Save Artifacts
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_FILE)
        
        # Save metadata (text, source) so we can retrieve it later
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
            
        print(f"✅ Index saved to {INDEX_FILE}")
        print(f"✅ Metadata saved to {METADATA_FILE}")
    else:
        print("⚠️ No data was processed.")

if __name__ == "__main__":
    build_pipeline()