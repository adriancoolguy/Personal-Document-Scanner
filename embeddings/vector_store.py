import numpy as np
from typing import List, Dict, Tuple, Optional
from .embedder import TextEmbedder
from sklearn.neighbors import NearestNeighbors

class VectorStore:
    def __init__(self, api_key: str):
        """Initialize the vector store with FAISS index and text embedder."""
        self.embedder = TextEmbedder(api_key=api_key)
        self.nn = None
        self.embeddings = None
        self.text_chunks = []
        self.original_rows = []
    
    def build_index(self, text_chunks: List[str], original_rows: List[Dict], progress_callback=None) -> None:
        """Build index from text chunks and store original rows."""
        if not text_chunks:
            raise ValueError("No text chunks provided")
        if progress_callback:
            progress_callback(0, len(text_chunks), "Starting embedding generation...")
        embeddings = self.embedder.get_embeddings(
            text_chunks,
            progress_callback=(lambda done, total: progress_callback(done, total, "Embedding...") if progress_callback else None)
        )
        if progress_callback:
            progress_callback(len(text_chunks), len(text_chunks), "Fitting NearestNeighbors index...")
        self.nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nn.fit(embeddings)
        self.embeddings = embeddings
        self.text_chunks = text_chunks
        self.original_rows = original_rows
        if progress_callback:
            progress_callback(len(text_chunks), len(text_chunks), "Index built!")
    
    def query(self, question: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Query the index with a question and return top-k matches."""
        if self.nn is None or self.embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Get question embedding
        question_embedding = self.embedder.get_embeddings([question])[0].reshape(1, -1)
        
        # Search the index
        distances, indices = self.nn.kneighbors(question_embedding, n_neighbors=top_k)
        
        # Return results with original rows and distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.original_rows):  # Ensure index is valid
                results.append((self.original_rows[idx], float(distance)))
        
        return results 