"""Base class for memory retrieval strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agentmemory.core.memory_entry import MemoryEntry


class BaseRetriever(ABC):
    """Abstract base class for memory retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """Retrieve memories based on query and filters."""
        pass
    
    def rerank(
        self,
        memories: List[MemoryEntry],
        query: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Rerank memories based on additional criteria."""
        # Default implementation returns as-is
        if limit:
            return memories[:limit]
        return memories