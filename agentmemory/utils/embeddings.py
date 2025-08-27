"""Embedding utilities for memory system."""

from typing import List, Optional, Union

import numpy as np


class EmbeddingManager:
    """Manager for creating and handling embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding manager."""
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            # Fallback for testing without sentence-transformers
            self._model = "mock"
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        if self._model == "mock":
            # Mock embeddings for testing
            return np.random.randn(len(texts), 384)
        
        return self._model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> List[float]:
        """Encode a single text into embedding."""
        embedding = self.encode(text)[0]
        return embedding.tolist()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(
        self,
        query_embedding: List[float],
        embeddings: List[List[float]]
    ) -> List[float]:
        """Calculate similarities between query and multiple embeddings."""
        query = np.array(query_embedding)
        vectors = np.array(embeddings)
        
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        # Normalize all vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors_norm = vectors / norms
        
        # Calculate cosine similarities
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities.tolist()


class TokenCounter:
    """Utility for counting tokens in text."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize token counter."""
        self.model = model
        self._encoding = None
    
    def _load_encoding(self) -> None:
        """Load tokenizer encoding."""
        if self._encoding is not None:
            return
        
        try:
            import tiktoken
            self._encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            # Fallback to approximate counting
            self._encoding = None
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        self._load_encoding()
        
        if self._encoding is None:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4
        
        return len(self._encoding.encode(text))
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        self._load_encoding()
        
        if self._encoding is None:
            # Approximate truncation
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars - 3] + "..."
        
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens - 1]
        return self._encoding.decode(truncated_tokens) + "..."