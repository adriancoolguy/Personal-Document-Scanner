import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any, Optional

class VectorStore:
    def __init__(self, model: Optional[SentenceTransformer] = None):
        """Initialize the vector store with an optional pre-loaded model."""
        self.model = model
        self.embeddings = None
        self.original_data = None
        
    def build_index(self, texts: List[str], original_data: List[Dict[str, Any]]) -> None:
        """Build the search index from texts and their corresponding data."""
        if self.model is None:
            raise ValueError("Model must be initialized before building index")
            
        # Get embeddings for all texts
        self.embeddings = self.model.encode(texts, show_progress_bar=False)
        self.original_data = original_data
        
    def query(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Query the index and return the most similar items with their scores."""
        if self.embeddings is None or self.original_data is None:
            raise ValueError("Index must be built before querying")
            
        # Get query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            score = float(similarities[idx])  # Convert to float for JSON serialization
            results.append((self.original_data[idx], score))
            
        return results

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Get the current embeddings for caching."""
        return self.embeddings

    def set_embeddings(self, embeddings: np.ndarray, original_data: List[Dict[str, Any]]) -> None:
        """Set embeddings and original data from cache."""
        self.embeddings = embeddings
        self.original_data = original_data 