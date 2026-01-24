import fitz  # PyMuPDF
from typing import List, Dict
# FIXED IMPORTS for LlamaIndex 0.9.x
from llama_index.node_parser import SentenceSplitter
from llama_index.schema import Document
import os

class PDFProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_and_chunk(self, pdf_path: str) -> List[Dict]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    doc_text += page.get_text()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return []

        document = Document(text=doc_text, metadata={"source": pdf_path})
        nodes = self.splitter.get_nodes_from_documents([document])

        chunks = []
        for i, node in enumerate(nodes):
            chunks.append({
                "id": f"{os.path.basename(pdf_path)}_chunk_{i}",
                "text": node.text,
                "source": pdf_path,
                "chunk_index": i
            })
            
        return chunks
