import numpy as np
from typing import List, Optional

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', api_key: Optional[str] = None, openai_model: str = 'text-embedding-3-small'):
        self.api_key = api_key
        self.openai_model = openai_model
        if api_key:
            import openai
            self.openai = openai
            self.openai.api_key = api_key
            self.use_openai = True
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.use_openai = False
    
    def get_embeddings(self, text_list: List[str], progress_callback=None) -> np.ndarray:
        """Convert a list of text strings into embeddings."""
        if not text_list:
            return np.array([])
        
        if self.use_openai:
            batch_size = 10  # Increased batch size for faster embedding if API quota allows
            embeddings = []
            total = len(text_list)
            for i in range(0, total, batch_size):
                batch = text_list[i:i+batch_size]
                response = self.openai.embeddings.create(
                    input=batch,
                    model=self.openai_model
                )
                batch_embeddings = [np.array(d.embedding) for d in response.data]
                embeddings.extend(batch_embeddings)
                if progress_callback:
                    progress_callback(min(i+batch_size, total), total)
            return np.vstack(embeddings)
        else:
            if progress_callback:
                progress_callback(len(text_list), len(text_list))
            embeddings = self.model.encode(text_list, show_progress_bar=True)
            return embeddings 