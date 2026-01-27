from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Auto-detect Apple Metal (MPS), CUDA, or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"🚀 Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        The RAG system calls this method specifically. 
        We pass the call directly to the internal model.
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        return embeddings

    # Keeping this for safety in case other parts of your code use it
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size)