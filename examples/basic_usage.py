"""Basic usage example of AgentMemory."""

from agentmemory import MemoryManager


def main():
    """Demonstrate basic usage of AgentMemory."""
    print("AgentMemory Basic Usage Example\n")
    
    # Initialize memory manager
    manager = MemoryManager(
        short_term_capacity=5,
        long_term_capacity=100,
        consolidation_threshold=3
    )
    
    # Example 1: Adding memories
    print("1. Adding memories to different stores:")
    
    # Add to short-term memory
    st_memory = manager.add(
        "User asked about weather in San Francisco",
        memory_type="short_term",
        tags=["weather", "query"],
        agent_id="assistant_001",
        session_id="session_123"
    )
    print(f"   Added to short-term: {st_memory.id}")
    
    # Add important fact to long-term memory
    lt_memory = manager.add(
        "The user prefers Python for data science projects",
        memory_type="long_term",
        importance=0.9,
        tags=["preference", "python", "data_science"]
    )
    print(f"   Added to long-term: {lt_memory.id}")
    
    # Add episodic memory
    ep_memory = manager.add(
        "User completed tutorial on machine learning basics",
        memory_type="episodic",
        episode_id="tutorial_ml_001",
        importance=0.7
    )
    print(f"   Added to episodic: {ep_memory.id}")
    
    # Add semantic knowledge
    sem_memory = manager.add(
        "Python is a high-level programming language known for its simplicity",
        memory_type="semantic",
        concepts=["python", "programming_language"],
        importance=0.8
    )
    print(f"   Added to semantic: {sem_memory.id}")
    
    # Add procedural knowledge
    proc_memory = manager.add(
        {
            "procedure": "Installing Python packages",
            "steps": [
                "Open terminal",
                "Activate virtual environment",
                "Run pip install <package_name>"
            ]
        },
        memory_type="procedural",
        procedure_name="install_python_package",
        skill_level=0.9
    )
    print(f"   Added to procedural: {proc_memory.id}\n")
    
    # Example 2: Retrieving memories
    print("2. Retrieving memories:")
    
    # Retrieve all recent memories
    all_memories = manager.retrieve(limit=3)
    print(f"   Retrieved {len(all_memories)} recent memories")
    for mem in all_memories:
        print(f"   - [{mem.memory_type}] {mem.content[:50]}...")
    
    # Retrieve from specific memory types
    print("\n   Short-term memories only:")
    st_memories = manager.retrieve(memory_types=["short_term"], limit=5)
    for mem in st_memories:
        print(f"   - {mem.content[:50]}...")
    
    # Example 3: Creating associations
    print("\n3. Creating associations between memories:")
    association_created = manager.create_association(
        lt_memory.id,
        sem_memory.id,
        relation_type="related_to"
    )
    print(f"   Association created: {association_created}")
    
    # Example 4: Memory consolidation
    print("\n4. Testing memory consolidation:")
    
    # Add more short-term memories to trigger consolidation
    manager.add("Query 1", memory_type="short_term", importance=0.2)
    manager.add("Query 2", memory_type="short_term", importance=0.8)
    manager.add("Query 3", memory_type="short_term", importance=0.6)
    
    print("   Consolidation triggered after threshold")
    stats = manager.consolidate_memories()
    print(f"   Consolidation results: {stats}")
    
    # Example 5: Getting statistics
    print("\n5. Memory system statistics:")
    stats = manager.get_statistics()
    print(f"   Total memories: {stats['total_memories']}")
    print("   By type:")
    for mem_type, count in stats["by_type"].items():
        print(f"   - {mem_type}: {count}")
    
    # Example 6: Saving and loading
    print("\n6. Saving memories to file:")
    manager.save("memories_backup.json")
    print("   Saved to memories_backup.json")
    
    # Create new manager and load
    new_manager = MemoryManager()
    new_manager.load("memories_backup.json")
    print("   Loaded into new manager")
    
    loaded_stats = new_manager.get_statistics()
    print(f"   Loaded {loaded_stats['total_memories']} memories")
    
    print("\n✅ Basic usage example completed!")


if __name__ == "__main__":
    main()