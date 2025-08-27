"""Semantic retrieval using embeddings."""

from typing import Any, Dict, List, Optional

from agentmemory.core.memory_entry import MemoryEntry
from agentmemory.retrievers.base import BaseRetriever
from agentmemory.stores.vector import VectorStore


class SemanticRetriever(BaseRetriever):
    """Retriever using semantic similarity search."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize with a vector store."""
        self.vector_store = vector_store
        self._embedding_model = None
    
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """Retrieve memories using semantic search."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search in vector store
        results = self.vector_store.search(
            {"embedding": query_embedding},
            limit=limit * 2  # Get more for filtering
        )
        
        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        # Return limited results
        return results[:limit]
    
    def retrieve_with_scores(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[tuple[MemoryEntry, float]]:
        """Retrieve memories with similarity scores."""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        if not query_embedding:
            return []
        
        # Search with scores
        results_with_scores = self.vector_store.search_with_scores(
            query_embedding,
            limit=limit * 2,
            threshold=threshold
        )
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for memory, score in results_with_scores:
                if self._matches_filters(memory, filters):
                    filtered_results.append((memory, score))
            results_with_scores = filtered_results
        
        # Return limited results
        return results_with_scores[:limit]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        # Initialize embedding model if needed
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to random embedding for demo
                import numpy as np
                return np.random.randn(384).tolist()
        
        # Generate embedding
        try:
            embedding = self._embedding_model.encode(text).tolist()
            return embedding
        except Exception:
            return None
    
    def _apply_filters(
        self,
        memories: List[MemoryEntry],
        filters: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """Apply filters to memory list."""
        filtered = []
        for memory in memories:
            if self._matches_filters(memory, filters):
                filtered.append(memory)
        return filtered
    
    def _matches_filters(
        self,
        memory: MemoryEntry,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if memory matches all filters."""
        for key, value in filters.items():
            if key == "memory_type" and memory.memory_type != value:
                return False
            elif key == "agent_id" and memory.metadata.agent_id != value:
                return False
            elif key == "session_id" and memory.metadata.session_id != value:
                return False
            elif key == "min_importance" and memory.metadata.importance_score < value:
                return False
            elif key == "max_importance" and memory.metadata.importance_score > value:
                return False
            elif key == "tags":
                if isinstance(value, str):
                    if value not in memory.metadata.tags:
                        return False
                elif isinstance(value, list):
                    if not any(tag in memory.metadata.tags for tag in value):
                        return False
        return True