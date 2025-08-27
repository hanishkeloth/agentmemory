"""Different types of memory implementations."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from agentmemory.core.memory_entry import MemoryEntry, MemoryMetadata


class BaseMemory(ABC):
    """Abstract base class for different memory types."""
    
    def __init__(self, capacity: Optional[int] = None):
        self.capacity = capacity
        self.memories: Dict[UUID, MemoryEntry] = {}
    
    @abstractmethod
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add a new memory."""
        pass
    
    @abstractmethod
    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memories based on query."""
        pass
    
    def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        return self.memories.get(memory_id)
    
    def update(self, memory_id: UUID, content: Optional[Union[str, Dict[str, Any]]] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory."""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        if content is not None:
            memory.content = content
        if metadata is not None:
            for key, value in metadata.items():
                setattr(memory.metadata, key, value)
        return True
    
    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
    
    def size(self) -> int:
        """Get the number of memories."""
        return len(self.memories)


class ShortTermMemory(BaseMemory):
    """Short-term memory with limited capacity and FIFO eviction."""
    
    def __init__(self, capacity: int = 10, ttl_seconds: int = 300):
        super().__init__(capacity)
        self.ttl_seconds = ttl_seconds
        self.memory_queue: deque = deque(maxlen=capacity)
    
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add a memory to short-term storage."""
        metadata = MemoryMetadata(
            importance_score=kwargs.get("importance", 0.3),
            tags=kwargs.get("tags", []),
            source=kwargs.get("source"),
            agent_id=kwargs.get("agent_id"),
            session_id=kwargs.get("session_id"),
        )
        
        entry = MemoryEntry(
            content=content,
            memory_type="short_term",
            metadata=metadata,
            embedding=kwargs.get("embedding")
        )
        
        # Remove expired memories
        self._cleanup_expired()
        
        # Handle capacity limit
        if self.capacity and len(self.memory_queue) >= self.capacity:
            oldest_id = self.memory_queue[0]
            if oldest_id in self.memories:
                del self.memories[oldest_id]
        
        self.memories[entry.id] = entry
        self.memory_queue.append(entry.id)
        return entry
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve recent memories."""
        self._cleanup_expired()
        
        memories = list(self.memories.values())
        memories.sort(key=lambda m: m.metadata.timestamp, reverse=True)
        
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def _cleanup_expired(self) -> None:
        """Remove memories that have exceeded TTL."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
        expired_ids = [
            mid for mid, memory in self.memories.items()
            if memory.metadata.timestamp < cutoff_time
        ]
        for mid in expired_ids:
            self.delete(mid)
            if mid in self.memory_queue:
                self.memory_queue.remove(mid)


class LongTermMemory(BaseMemory):
    """Long-term memory with importance-based retention."""
    
    def __init__(self, capacity: Optional[int] = 1000, importance_threshold: float = 0.4):
        super().__init__(capacity)
        self.importance_threshold = importance_threshold
    
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add a memory to long-term storage if important enough."""
        importance = kwargs.get("importance", 0.5)
        
        if importance < self.importance_threshold:
            raise ValueError(f"Importance {importance} below threshold {self.importance_threshold}")
        
        metadata = MemoryMetadata(
            importance_score=importance,
            decay_rate=kwargs.get("decay_rate", 0.001),
            tags=kwargs.get("tags", []),
            source=kwargs.get("source"),
            agent_id=kwargs.get("agent_id"),
            session_id=kwargs.get("session_id"),
        )
        
        entry = MemoryEntry(
            content=content,
            memory_type="long_term",
            metadata=metadata,
            embedding=kwargs.get("embedding")
        )
        
        # Handle capacity with importance-based eviction
        if self.capacity and self.size() >= self.capacity:
            self._evict_least_important()
        
        self.memories[entry.id] = entry
        return entry
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memories by relevance."""
        memories = list(self.memories.values())
        
        # Sort by relevance score
        memories.sort(key=lambda m: m.calculate_relevance(), reverse=True)
        
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def _evict_least_important(self) -> None:
        """Remove the least important memory."""
        if not self.memories:
            return
        
        least_important = min(
            self.memories.values(),
            key=lambda m: m.calculate_relevance()
        )
        self.delete(least_important.id)


class EpisodicMemory(BaseMemory):
    """Memory for specific episodes or events with temporal organization."""
    
    def __init__(self, capacity: Optional[int] = 500):
        super().__init__(capacity)
        self.episodes: Dict[str, List[UUID]] = {}
    
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add an episodic memory."""
        episode_id = kwargs.get("episode_id", "default")
        
        metadata = MemoryMetadata(
            importance_score=kwargs.get("importance", 0.6),
            tags=kwargs.get("tags", []),
            source=kwargs.get("source"),
            agent_id=kwargs.get("agent_id"),
            session_id=kwargs.get("session_id"),
            custom_metadata={"episode_id": episode_id}
        )
        
        entry = MemoryEntry(
            content=content,
            memory_type="episodic",
            metadata=metadata,
            embedding=kwargs.get("embedding")
        )
        
        # Link to episode
        if episode_id not in self.episodes:
            self.episodes[episode_id] = []
        self.episodes[episode_id].append(entry.id)
        
        # Link to previous memory in episode
        if len(self.episodes[episode_id]) > 1:
            prev_id = self.episodes[episode_id][-2]
            entry.metadata.parent_memory_id = prev_id
            if prev_id in self.memories:
                self.memories[prev_id].metadata.child_memory_ids.append(entry.id)
        
        self.memories[entry.id] = entry
        return entry
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10,
                episode_id: Optional[str] = None) -> List[MemoryEntry]:
        """Retrieve episodic memories."""
        if episode_id and episode_id in self.episodes:
            memory_ids = self.episodes[episode_id]
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            memories = list(self.memories.values())
        
        memories.sort(key=lambda m: m.metadata.timestamp, reverse=True)
        
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def get_episode(self, episode_id: str) -> List[MemoryEntry]:
        """Get all memories from a specific episode."""
        if episode_id not in self.episodes:
            return []
        
        memories = []
        for memory_id in self.episodes[episode_id]:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.update_access()
                memories.append(memory)
        
        return memories


class SemanticMemory(BaseMemory):
    """Memory for facts and conceptual knowledge with semantic relationships."""
    
    def __init__(self, capacity: Optional[int] = 2000):
        super().__init__(capacity)
        self.concepts: Dict[str, List[UUID]] = {}
        self.relationships: Dict[str, List[tuple[UUID, UUID]]] = {}
    
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add semantic memory with concepts."""
        concepts = kwargs.get("concepts", [])
        
        metadata = MemoryMetadata(
            importance_score=kwargs.get("importance", 0.7),
            decay_rate=kwargs.get("decay_rate", 0.0001),
            tags=concepts + kwargs.get("tags", []),
            source=kwargs.get("source"),
            agent_id=kwargs.get("agent_id"),
            custom_metadata={"concepts": concepts}
        )
        
        entry = MemoryEntry(
            content=content,
            memory_type="semantic",
            metadata=metadata,
            embedding=kwargs.get("embedding")
        )
        
        # Index by concepts
        for concept in concepts:
            if concept not in self.concepts:
                self.concepts[concept] = []
            self.concepts[concept].append(entry.id)
        
        self.memories[entry.id] = entry
        return entry
    
    def add_relationship(self, memory_id1: UUID, memory_id2: UUID, 
                        relation_type: str = "related") -> None:
        """Add a semantic relationship between memories."""
        if relation_type not in self.relationships:
            self.relationships[relation_type] = []
        
        self.relationships[relation_type].append((memory_id1, memory_id2))
        
        if memory_id1 in self.memories:
            self.memories[memory_id1].add_relation(relation_type, memory_id2)
        if memory_id2 in self.memories:
            self.memories[memory_id2].add_relation(f"inverse_{relation_type}", memory_id1)
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10,
                concepts: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Retrieve semantic memories by concepts."""
        if concepts:
            memory_ids = set()
            for concept in concepts:
                if concept in self.concepts:
                    memory_ids.update(self.concepts[concept])
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            memories = list(self.memories.values())
        
        memories.sort(key=lambda m: m.calculate_relevance(), reverse=True)
        
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def get_related(self, memory_id: UUID, relation_type: Optional[str] = None) -> List[MemoryEntry]:
        """Get memories related to a specific memory."""
        if memory_id not in self.memories:
            return []
        
        memory = self.memories[memory_id]
        related_ids = []
        
        if relation_type:
            related_ids = memory.metadata.relations.get(relation_type, [])
        else:
            for relations in memory.metadata.relations.values():
                related_ids.extend(relations)
        
        return [self.memories[mid] for mid in related_ids if mid in self.memories]


class ProceduralMemory(BaseMemory):
    """Memory for skills, procedures, and how-to knowledge."""
    
    def __init__(self, capacity: Optional[int] = 500):
        super().__init__(capacity)
        self.procedures: Dict[str, List[UUID]] = {}
        self.skills: Dict[str, float] = {}
    
    def add(self, content: Union[str, Dict[str, Any]], **kwargs) -> MemoryEntry:
        """Add procedural memory."""
        procedure_name = kwargs.get("procedure_name", "unknown")
        steps = kwargs.get("steps", [])
        skill_level = kwargs.get("skill_level", 0.5)
        
        metadata = MemoryMetadata(
            importance_score=kwargs.get("importance", 0.8),
            decay_rate=kwargs.get("decay_rate", 0.0001),
            tags=[procedure_name] + kwargs.get("tags", []),
            source=kwargs.get("source"),
            agent_id=kwargs.get("agent_id"),
            custom_metadata={
                "procedure_name": procedure_name,
                "steps": steps,
                "skill_level": skill_level,
                "execution_count": 0,
                "success_rate": 0.0
            }
        )
        
        entry = MemoryEntry(
            content=content,
            memory_type="procedural",
            metadata=metadata,
            embedding=kwargs.get("embedding")
        )
        
        # Index by procedure
        if procedure_name not in self.procedures:
            self.procedures[procedure_name] = []
        self.procedures[procedure_name].append(entry.id)
        
        # Track skill level
        self.skills[procedure_name] = skill_level
        
        self.memories[entry.id] = entry
        return entry
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10,
                procedure_name: Optional[str] = None) -> List[MemoryEntry]:
        """Retrieve procedural memories."""
        if procedure_name and procedure_name in self.procedures:
            memory_ids = self.procedures[procedure_name]
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            memories = list(self.memories.values())
        
        # Sort by skill level and success rate
        def score(m):
            skill = m.metadata.custom_metadata.get("skill_level", 0)
            success = m.metadata.custom_metadata.get("success_rate", 0)
            return skill * 0.6 + success * 0.4
        
        memories.sort(key=score, reverse=True)
        
        for memory in memories[:limit]:
            memory.update_access()
        
        return memories[:limit]
    
    def update_execution(self, memory_id: UUID, success: bool) -> None:
        """Update execution statistics for a procedure."""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        metadata = memory.metadata.custom_metadata
        
        metadata["execution_count"] = metadata.get("execution_count", 0) + 1
        executions = metadata["execution_count"]
        current_rate = metadata.get("success_rate", 0.0)
        
        # Update success rate with moving average
        if success:
            new_rate = (current_rate * (executions - 1) + 1.0) / executions
        else:
            new_rate = (current_rate * (executions - 1)) / executions
        
        metadata["success_rate"] = new_rate
        
        # Update skill level
        procedure_name = metadata.get("procedure_name")
        if procedure_name:
            self.skills[procedure_name] = min(1.0, self.skills.get(procedure_name, 0.5) + 0.01 if success else max(0.0, self.skills.get(procedure_name, 0.5) - 0.01))