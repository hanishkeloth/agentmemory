"""Tests for MemoryManager."""

import json
import tempfile
from pathlib import Path

import pytest

from agentmemory import MemoryManager


class TestMemoryManager:
    """Test suite for MemoryManager."""
    
    def test_initialization(self):
        """Test memory manager initialization."""
        manager = MemoryManager()
        assert "short_term" in manager.memories
        assert "long_term" in manager.memories
        assert "episodic" in manager.memories
        assert "semantic" in manager.memories
        assert "procedural" in manager.memories
    
    def test_add_memory(self):
        """Test adding memories to different types."""
        manager = MemoryManager()
        
        # Add to short-term
        entry1 = manager.add("Test memory 1", memory_type="short_term")
        assert entry1 is not None
        assert entry1.content == "Test memory 1"
        assert entry1.memory_type == "short_term"
        
        # Add to long-term with high importance
        entry2 = manager.add("Important memory", memory_type="long_term", importance=0.8)
        assert entry2 is not None
        assert entry2.metadata.importance_score == 0.8
    
    def test_retrieve_memory(self):
        """Test retrieving memories."""
        manager = MemoryManager()
        
        # Add some memories
        manager.add("Memory A", memory_type="short_term", tags=["test"])
        manager.add("Memory B", memory_type="short_term", tags=["demo"])
        manager.add("Memory C", memory_type="long_term", importance=0.9)
        
        # Retrieve all
        memories = manager.retrieve(limit=10)
        assert len(memories) > 0
        
        # Retrieve from specific types
        short_term_memories = manager.retrieve(memory_types=["short_term"])
        assert all(m.memory_type == "short_term" for m in short_term_memories)
    
    def test_consolidation(self):
        """Test memory consolidation."""
        manager = MemoryManager(consolidation_threshold=2)
        
        # Add memories to trigger consolidation
        manager.add("Memory 1", memory_type="short_term", importance=0.8)
        manager.add("Memory 2", memory_type="short_term", importance=0.3)
        
        # Check consolidation occurred
        stats = manager.get_statistics()
        assert stats["consolidation_counter"] == 0  # Reset after consolidation
    
    def test_associations(self):
        """Test creating associations between memories."""
        manager = MemoryManager()
        
        # Add two memories
        entry1 = manager.add("Fact 1", memory_type="semantic")
        entry2 = manager.add("Fact 2", memory_type="semantic")
        
        # Create association
        success = manager.create_association(entry1.id, entry2.id, "related_to")
        assert success
        
        # Check relations
        memory1 = manager.get_memory(entry1.id)
        assert entry2.id in memory1.metadata.relations.get("related_to", [])
    
    def test_update_memory(self):
        """Test updating memory content and metadata."""
        manager = MemoryManager()
        
        # Add a memory
        entry = manager.add("Original content", memory_type="short_term")
        
        # Update content
        success = manager.update_memory(entry.id, content="Updated content")
        assert success
        
        # Verify update
        updated = manager.get_memory(entry.id)
        assert updated.content == "Updated content"
    
    def test_delete_memory(self):
        """Test deleting memories."""
        manager = MemoryManager()
        
        # Add and delete a memory
        entry = manager.add("To be deleted", memory_type="short_term")
        success = manager.delete_memory(entry.id)
        assert success
        
        # Verify deletion
        assert manager.get_memory(entry.id) is None
    
    def test_save_and_load(self):
        """Test saving and loading memories."""
        manager1 = MemoryManager()
        
        # Add some memories
        entry1 = manager1.add("Memory 1", memory_type="short_term")
        entry2 = manager1.add("Memory 2", memory_type="long_term", importance=0.8)
        
        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
        
        manager1.save(filepath)
        
        # Load into new manager
        manager2 = MemoryManager()
        manager2.load(filepath)
        
        # Verify memories loaded
        loaded1 = manager2.get_memory(entry1.id)
        loaded2 = manager2.get_memory(entry2.id)
        
        assert loaded1 is not None
        assert loaded1.content == "Memory 1"
        assert loaded2 is not None
        assert loaded2.content == "Memory 2"
        
        # Cleanup
        filepath.unlink()
    
    def test_statistics(self):
        """Test getting memory statistics."""
        manager = MemoryManager()
        
        # Add memories to different stores
        manager.add("ST1", memory_type="short_term")
        manager.add("ST2", memory_type="short_term")
        manager.add("LT1", memory_type="long_term", importance=0.8)
        
        stats = manager.get_statistics()
        assert stats["total_memories"] == 3
        assert stats["by_type"]["short_term"] == 2
        assert stats["by_type"]["long_term"] == 1
    
    def test_clear_all(self):
        """Test clearing all memories."""
        manager = MemoryManager()
        
        # Add some memories
        manager.add("Memory 1", memory_type="short_term")
        manager.add("Memory 2", memory_type="long_term", importance=0.8)
        
        # Clear all
        manager.clear_all()
        
        # Verify all cleared
        stats = manager.get_statistics()
        assert stats["total_memories"] == 0