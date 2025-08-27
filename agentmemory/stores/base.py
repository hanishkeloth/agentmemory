"""Base class for memory storage implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from agentmemory.core.memory_entry import MemoryEntry


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def add(self, memory: MemoryEntry) -> bool:
        """Add a memory to the store."""
        pass
    
    @abstractmethod
    def get(self, memory_id: UUID) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        pass
    
    @abstractmethod
    def update(self, memory: MemoryEntry) -> bool:
        """Update an existing memory."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory from the store."""
        pass
    
    @abstractmethod
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for memories based on criteria."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memories from the store."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of memories in the store."""
        pass
    
    def batch_add(self, memories: List[MemoryEntry]) -> int:
        """Add multiple memories at once."""
        count = 0
        for memory in memories:
            if self.add(memory):
                count += 1
        return count
    
    def batch_delete(self, memory_ids: List[UUID]) -> int:
        """Delete multiple memories at once."""
        count = 0
        for memory_id in memory_ids:
            if self.delete(memory_id):
                count += 1
        return count