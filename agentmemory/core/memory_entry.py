"""Memory entry and metadata structures."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryMetadata(BaseModel):
    """Metadata associated with a memory entry."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.01, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_memory_id: Optional[UUID] = None
    child_memory_ids: List[UUID] = Field(default_factory=list)
    relations: Dict[str, List[UUID]] = Field(default_factory=dict)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(BaseModel):
    """A single memory entry in the system."""
    
    id: UUID = Field(default_factory=uuid4)
    content: Union[str, Dict[str, Any]]
    embedding: Optional[List[float]] = None
    memory_type: str
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    
    def update_access(self) -> None:
        """Update access metadata when memory is retrieved."""
        self.metadata.access_count += 1
        self.metadata.last_accessed = datetime.utcnow()
    
    def calculate_relevance(self, base_score: float = 0.5) -> float:
        """Calculate relevance score with time decay."""
        if self.metadata.last_accessed:
            time_since_access = (datetime.utcnow() - self.metadata.last_accessed).total_seconds()
            decay_factor = 1.0 / (1.0 + self.metadata.decay_rate * time_since_access / 3600)
        else:
            decay_factor = 1.0
        
        importance_weight = self.metadata.importance_score
        access_weight = min(1.0, self.metadata.access_count / 100)
        
        return base_score * decay_factor * (0.5 + 0.3 * importance_weight + 0.2 * access_weight)
    
    def add_relation(self, relation_type: str, memory_id: UUID) -> None:
        """Add a relation to another memory."""
        if relation_type not in self.metadata.relations:
            self.metadata.relations[relation_type] = []
        if memory_id not in self.metadata.relations[relation_type]:
            self.metadata.relations[relation_type].append(memory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type,
            "metadata": self.metadata.model_dump()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        if "id" in data and isinstance(data["id"], str):
            data["id"] = UUID(data["id"])
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = MemoryMetadata(**data["metadata"])
        return cls(**data)