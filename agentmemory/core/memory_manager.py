"""Memory manager for coordinating different memory types."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from agentmemory.core.memory_entry import MemoryEntry
from agentmemory.core.memory_types import (
    EpisodicMemory,
    LongTermMemory,
    ProceduralMemory,
    SemanticMemory,
    ShortTermMemory,
)
from agentmemory.retrievers.semantic import SemanticRetriever
from agentmemory.stores.vector import VectorStore


class MemoryManager:
    """Central manager for all memory types and operations."""
    
    def __init__(
        self,
        enable_short_term: bool = True,
        enable_long_term: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_procedural: bool = True,
        enable_vector_store: bool = True,
        vector_dimension: int = 384,
        short_term_capacity: int = 10,
        long_term_capacity: int = 1000,
        episodic_capacity: int = 500,
        semantic_capacity: int = 2000,
        procedural_capacity: int = 500,
    ):
        """Initialize memory manager with configurable memory types."""
        self.memories = {}
        
        if enable_short_term:
            self.memories["short_term"] = ShortTermMemory(capacity=short_term_capacity)
        
        if enable_long_term:
            self.memories["long_term"] = LongTermMemory(capacity=long_term_capacity)
        
        if enable_episodic:
            self.memories["episodic"] = EpisodicMemory(capacity=episodic_capacity)
        
        if enable_semantic:
            self.memories["semantic"] = SemanticMemory(capacity=semantic_capacity)
        
        if enable_procedural:
            self.memories["procedural"] = ProceduralMemory(capacity=procedural_capacity)
        
        self.vector_store = None
        self.semantic_retriever = None
        if enable_vector_store:
            self.vector_store = VectorStore(dimension=vector_dimension)
            self.semantic_retriever = SemanticRetriever(self.vector_store)
        
        self.consolidation_threshold = 5
        self.consolidation_counter = 0
    
    def add(
        self,
        content: Union[str, Dict[str, Any]],
        memory_type: str = "short_term",
        auto_consolidate: bool = True,
        **kwargs
    ) -> MemoryEntry:
        """Add a memory to the specified memory type."""
        if memory_type not in self.memories:
            raise ValueError(f"Memory type '{memory_type}' not enabled or invalid")
        
        memory = self.memories[memory_type]
        entry = memory.add(content, **kwargs)
        
        # Add to vector store if enabled and embedding provided
        if self.vector_store and "embedding" in kwargs and kwargs["embedding"]:
            self.vector_store.add(entry)
        
        # Auto-consolidate if enabled
        if auto_consolidate:
            self.consolidation_counter += 1
            if self.consolidation_counter >= self.consolidation_threshold:
                self.consolidate_memories()
                self.consolidation_counter = 0
        
        return entry
    
    def retrieve(
        self,
        query: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        use_semantic_search: bool = True,
        **kwargs
    ) -> List[MemoryEntry]:
        """Retrieve memories from specified types."""
        if memory_types is None:
            memory_types = list(self.memories.keys())
        
        all_memories = []
        
        # Use semantic search if available and requested
        if use_semantic_search and self.semantic_retriever and query:
            semantic_results = self.semantic_retriever.retrieve(query, limit=limit)
            all_memories.extend(semantic_results)
        else:
            # Retrieve from each memory type
            for memory_type in memory_types:
                if memory_type in self.memories:
                    memories = self.memories[memory_type].retrieve(query, limit=limit, **kwargs)
                    all_memories.extend(memories)
        
        # Remove duplicates and sort by relevance
        unique_memories = {}
        for memory in all_memories:
            if memory.id not in unique_memories:
                unique_memories[memory.id] = memory
        
        memories = list(unique_memories.values())
        memories.sort(key=lambda m: m.calculate_relevance(), reverse=True)
        
        return memories[:limit]
    
    def consolidate_memories(self) -> Dict[str, int]:
        """Consolidate short-term memories into long-term storage."""
        consolidation_stats = {
            "promoted_to_long_term": 0,
            "promoted_to_semantic": 0,
            "promoted_to_episodic": 0,
            "discarded": 0,
        }
        
        if "short_term" not in self.memories:
            return consolidation_stats
        
        short_term = self.memories["short_term"]
        memories_to_consolidate = short_term.retrieve(limit=100)
        
        for memory in memories_to_consolidate:
            importance = memory.metadata.importance_score
            
            # Decide where to consolidate based on importance and characteristics
            if importance >= 0.7:
                # High importance -> long-term
                if "long_term" in self.memories:
                    try:
                        self.memories["long_term"].add(
                            memory.content,
                            importance=importance,
                            embedding=memory.embedding,
                            tags=memory.metadata.tags,
                            source=memory.metadata.source,
                            agent_id=memory.metadata.agent_id,
                        )
                        consolidation_stats["promoted_to_long_term"] += 1
                    except ValueError:
                        pass
            
            elif importance >= 0.5:
                # Medium importance with concepts -> semantic
                if "semantic" in self.memories and memory.metadata.tags:
                    self.memories["semantic"].add(
                        memory.content,
                        concepts=memory.metadata.tags[:3],
                        importance=importance,
                        embedding=memory.embedding,
                        source=memory.metadata.source,
                        agent_id=memory.metadata.agent_id,
                    )
                    consolidation_stats["promoted_to_semantic"] += 1
            
            elif importance >= 0.3 and memory.metadata.session_id:
                # Episode-related -> episodic
                if "episodic" in self.memories:
                    self.memories["episodic"].add(
                        memory.content,
                        episode_id=memory.metadata.session_id,
                        importance=importance,
                        embedding=memory.embedding,
                        tags=memory.metadata.tags,
                        source=memory.metadata.source,
                        agent_id=memory.metadata.agent_id,
                    )
                    consolidation_stats["promoted_to_episodic"] += 1
            else:
                consolidation_stats["discarded"] += 1
        
        # Clear consolidated memories from short-term
        short_term.clear()
        
        return consolidation_stats
    
    def create_association(
        self,
        memory_id1: UUID,
        memory_id2: UUID,
        relation_type: str = "related"
    ) -> bool:
        """Create an association between two memories."""
        memory1 = self.get_memory(memory_id1)
        memory2 = self.get_memory(memory_id2)
        
        if not memory1 or not memory2:
            return False
        
        memory1.add_relation(relation_type, memory_id2)
        memory2.add_relation(f"inverse_{relation_type}", memory_id1)
        
        # If semantic memory, update its relationships
        if "semantic" in self.memories:
            semantic = self.memories["semantic"]
            if memory1.memory_type == "semantic":
                semantic.add_relationship(memory_id1, memory_id2, relation_type)
        
        return True
    
    def get_memory(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Get a specific memory by ID from any memory type."""
        for memory_store in self.memories.values():
            memory = memory_store.get(memory_id)
            if memory:
                return memory
        return None
    
    def update_memory(
        self,
        memory_id: UUID,
        content: Optional[Union[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a memory's content or metadata."""
        for memory_store in self.memories.values():
            if memory_store.update(memory_id, content, metadata):
                # Update in vector store if present
                if self.vector_store:
                    memory = memory_store.get(memory_id)
                    if memory and memory.embedding:
                        self.vector_store.update(memory)
                return True
        return False
    
    def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory from all stores."""
        deleted = False
        for memory_store in self.memories.values():
            if memory_store.delete(memory_id):
                deleted = True
        
        # Remove from vector store
        if self.vector_store:
            self.vector_store.delete(memory_id)
        
        return deleted
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save all memories to a file."""
        filepath = Path(filepath)
        
        data = {
            "memories": {},
            "vector_store": None,
        }
        
        # Save each memory type
        for memory_type, memory_store in self.memories.items():
            data["memories"][memory_type] = [
                memory.to_dict() for memory in memory_store.memories.values()
            ]
        
        # Save vector store if present
        if self.vector_store:
            data["vector_store"] = self.vector_store.to_dict()
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load memories from a file."""
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Clear existing memories
        for memory_store in self.memories.values():
            memory_store.clear()
        
        # Load each memory type
        for memory_type, memories_data in data.get("memories", {}).items():
            if memory_type in self.memories:
                memory_store = self.memories[memory_type]
                for memory_data in memories_data:
                    memory = MemoryEntry.from_dict(memory_data)
                    memory_store.memories[memory.id] = memory
        
        # Load vector store if present
        if self.vector_store and data.get("vector_store"):
            self.vector_store.from_dict(data["vector_store"])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        stats = {
            "total_memories": 0,
            "by_type": {},
            "consolidation_counter": self.consolidation_counter,
        }
        
        for memory_type, memory_store in self.memories.items():
            count = memory_store.size()
            stats["by_type"][memory_type] = count
            stats["total_memories"] += count
        
        if self.vector_store:
            stats["vector_store_size"] = len(self.vector_store.index)
        
        return stats
    
    def clear_all(self) -> None:
        """Clear all memories from all stores."""
        for memory_store in self.memories.values():
            memory_store.clear()
        
        if self.vector_store:
            self.vector_store.clear()
        
        self.consolidation_counter = 0