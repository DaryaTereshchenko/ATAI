"""
Query Embedder using Sentence Transformers.
Converts natural language queries into dense embeddings.
"""

import numpy as np
from typing import Optional


class QueryEmbedder:
    """Embeds natural language queries using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the query embedder.
        
        Args:
            model_name: Sentence transformer model name
                - "sentence-transformers/all-MiniLM-L6-v2" (default, 384 dim, fast)
                - "sentence-transformers/all-mpnet-base-v2" (768 dim, better quality)
                - "sentence-transformers/paraphrase-MiniLM-L6-v2" (384 dim, paraphrase)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"📥 Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully")
            print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a natural language query.
        
        Args:
            query: Natural language query string
            
        Returns:
            Query embedding as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Preprocess query
        query = self._preprocess_query(query)
        
        # Generate embedding
        embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        
        return embedding
    
    def embed_batch(self, queries: list) -> np.ndarray:
        """
        Embed multiple queries in batch.
        
        Args:
            queries: List of query strings
            
        Returns:
            Array of embeddings, shape (n_queries, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Preprocess queries
        queries = [self._preprocess_query(q) for q in queries]
        
        # Generate embeddings
        embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
        
        return embeddings
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query text before embedding.
        
        Args:
            query: Raw query string
            
        Returns:
            Preprocessed query string
        """
        # Basic preprocessing
        query = query.strip()
        
        # Remove extra whitespace
        import re
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.get_sentence_embedding_dimension()
