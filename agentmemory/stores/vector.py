"""Vector store for semantic memory search."""

import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from agentmemory.core.memory_entry import MemoryEntry
from agentmemory.stores.base import BaseMemoryStore


class VectorStore(BaseMemoryStore):
    """In-memory vector store using NumPy for similarity search."""
    
    def __init__(self, dimension: int = 384):
        """Initialize vector store with specified embedding dimension."""
        self.dimension = dimension
        self.index: Dict[UUID, MemoryEntry] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.memory_ids: List[UUID] = []
    
    def add(self, memory: MemoryEntry) -> bool:
        """Add a memory with embedding to the store."""
        if not memory.embedding or len(memory.embedding) != self.dimension:
            return False
        
        self.index[memory.id] = memory
        self.memory_ids.append(memory.id)
        
        # Add embedding to matrix
        embedding_array = np.array(memory.embedding, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = embedding_array.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_array])
        
        return True
    
    def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        return self.index.get(memory_id)
    
    def update(self, memory: MemoryEntry) -> bool:
        """Update an existing memory and its embedding."""
        if memory.id not in self.index:
            return False
        
        if not memory.embedding or len(memory.embedding) != self.dimension:
            return False
        
        # Update memory
        self.index[memory.id] = memory
        
        # Update embedding
        idx = self.memory_ids.index(memory.id)
        self.embeddings[idx] = np.array(memory.embedding, dtype=np.float32)
        
        return True
    
    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory from the store."""
        if memory_id not in self.index:
            return False
        
        # Remove from index
        del self.index[memory_id]
        
        # Remove from embeddings
        idx = self.memory_ids.index(memory_id)
        self.memory_ids.pop(idx)
        
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)
            if self.embeddings.shape[0] == 0:
                self.embeddings = None
        
        return True
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for similar memories using cosine similarity."""
        if "embedding" not in query or self.embeddings is None:
            return []
        
        query_embedding = np.array(query["embedding"], dtype=np.float32)
        
        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        # Return corresponding memories
        results = []
        for idx in top_indices:
            if idx < len(self.memory_ids):
                memory_id = self.memory_ids[idx]
                memory = self.index[memory_id]
                memory.update_access()
                results.append(memory)
        
        return results
    
    def search_with_scores(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories and return with similarity scores."""
        if self.embeddings is None:
            return []
        
        query_array = np.array(query_embedding, dtype=np.float32)
        
        # Calculate cosine similarity
        similarities = self._cosine_similarity(query_array, self.embeddings)
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        valid_similarities = similarities[valid_indices]
        
        # Sort by similarity
        sorted_indices = np.argsort(valid_similarities)[-limit:][::-1]
        
        # Return memories with scores
        results = []
        for idx in sorted_indices:
            original_idx = valid_indices[idx]
            if original_idx < len(self.memory_ids):
                memory_id = self.memory_ids[original_idx]
                memory = self.index[memory_id]
                memory.update_access()
                score = float(valid_similarities[idx])
                results.append((memory, score))
        
        return results
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        self.index.clear()
        self.memory_ids.clear()
        self.embeddings = None
    
    def size(self) -> int:
        """Get the number of memories in the store."""
        return len(self.index)
    
    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all embeddings."""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_norm = embeddings / norms
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        return similarities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vector store to dictionary for serialization."""
        return {
            "dimension": self.dimension,
            "memories": [memory.to_dict() for memory in self.index.values()],
            "memory_ids": [str(mid) for mid in self.memory_ids],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load vector store from dictionary."""
        self.clear()
        self.dimension = data.get("dimension", self.dimension)
        
        # Load memories
        for memory_data in data.get("memories", []):
            memory = MemoryEntry.from_dict(memory_data)
            self.index[memory.id] = memory
        
        # Load memory IDs
        self.memory_ids = [UUID(mid) for mid in data.get("memory_ids", [])]
        
        # Load embeddings
        embeddings_data = data.get("embeddings")
        if embeddings_data:
            self.embeddings = np.array(embeddings_data, dtype=np.float32)
        else:
            self.embeddings = None


class FaissVectorStore(BaseMemoryStore):
    """Vector store using FAISS for efficient similarity search."""
    
    def __init__(self, dimension: int = 384, index_type: str = "IndexFlatL2"):
        """Initialize FAISS vector store."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required for FaissVectorStore. Install with: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "IndexFlatL2":
            self.faiss_index = faiss.IndexFlatL2(dimension)
        elif index_type == "IndexFlatIP":
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexHNSW":
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.memories: Dict[int, MemoryEntry] = {}
        self.id_to_idx: Dict[UUID, int] = {}
        self.next_idx = 0
    
    def add(self, memory: MemoryEntry) -> bool:
        """Add a memory with embedding to the FAISS index."""
        if not memory.embedding or len(memory.embedding) != self.dimension:
            return False
        
        import faiss
        
        # Add to FAISS index
        embedding = np.array([memory.embedding], dtype=np.float32)
        self.faiss_index.add(embedding)
        
        # Store memory
        idx = self.next_idx
        self.memories[idx] = memory
        self.id_to_idx[memory.id] = idx
        self.next_idx += 1
        
        return True
    
    def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        idx = self.id_to_idx.get(memory_id)
        if idx is not None:
            return self.memories.get(idx)
        return None
    
    def update(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        if memory.id not in self.id_to_idx:
            return False
        
        idx = self.id_to_idx[memory.id]
        self.memories[idx] = memory
        
        # Note: FAISS doesn't support in-place updates of embeddings
        # For full update support, would need to rebuild index
        
        return True
    
    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory from the store."""
        if memory_id not in self.id_to_idx:
            return False
        
        idx = self.id_to_idx[memory_id]
        del self.memories[idx]
        del self.id_to_idx[memory_id]
        
        # Note: FAISS doesn't support deletion from index
        # Would need to rebuild index for true deletion
        
        return True
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for similar memories using FAISS."""
        if "embedding" not in query or self.faiss_index.ntotal == 0:
            return []
        
        # Prepare query embedding
        query_embedding = np.array([query["embedding"]], dtype=np.float32)
        
        # Search in FAISS
        distances, indices = self.faiss_index.search(query_embedding, min(limit, self.faiss_index.ntotal))
        
        # Return corresponding memories
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx in self.memories:
                memory = self.memories[idx]
                memory.update_access()
                results.append(memory)
        
        return results
    
    def clear(self) -> None:
        """Clear all memories from the store."""
        import faiss
        
        self.memories.clear()
        self.id_to_idx.clear()
        self.next_idx = 0
        
        # Reset FAISS index
        if self.index_type == "IndexFlatL2":
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexHNSW":
            self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)
    
    def size(self) -> int:
        """Get the number of memories in the store."""
        return len(self.memories)