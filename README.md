# AgentMemory 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/hanishkeloth/agentmemory)

**Language Versions**: [English](README.md) | [한국어](README.ko.md) | [Deutsch](README.de.md) | [日本語](README.ja.md) | [العربية](README.ar.md)

**AgentMemory** is an advanced memory management framework for AI agents, providing persistent, hierarchical, and semantic memory capabilities. It addresses the critical gap in current agentic AI development: robust memory and context management across agent sessions.

## 🚀 Key Features

### 🎯 Multiple Memory Types
- **Short-Term Memory**: FIFO buffer with TTL for immediate context
- **Long-Term Memory**: Importance-based retention for valuable information
- **Episodic Memory**: Temporal organization of event sequences
- **Semantic Memory**: Concept-based knowledge with relationships
- **Procedural Memory**: Skills and how-to knowledge with execution tracking

### 🔍 Advanced Retrieval
- **Semantic Search**: Vector similarity search using embeddings
- **Filtered Queries**: Retrieve by type, tags, importance, or custom metadata
- **Relationship Navigation**: Follow memory associations and relations
- **Time-Decay Scoring**: Automatic relevance calculation with decay

### 💾 Persistence & Storage
- **Multiple Backends**: In-memory, vector stores (NumPy/FAISS)
- **Save/Load**: JSON serialization for memory persistence
- **Batch Operations**: Efficient bulk add/delete operations

### 🔄 Memory Management
- **Auto-Consolidation**: Automatic promotion from short-term to long-term
- **Importance Scoring**: Configurable importance thresholds
- **Memory Associations**: Create relationships between memories
- **Statistics Tracking**: Monitor memory usage and patterns

## 📦 Installation

```bash
pip install agentmemory
```

For development:
```bash
git clone https://github.com/hanishkeloth/agentmemory.git
cd agentmemory
pip install -e ".[dev]"
```

## 🎓 Quick Start

```python
from agentmemory import MemoryManager

# Initialize memory manager
memory = MemoryManager()

# Add memories
memory.add(
    "User prefers Python for data science",
    memory_type="long_term",
    importance=0.9,
    tags=["preference", "python"]
)

# Retrieve relevant memories
memories = memory.retrieve(
    query="What programming language to use?",
    limit=5
)

# Create associations
memory.create_association(memory1_id, memory2_id, "related_to")

# Save for persistence
memory.save("agent_memories.json")
```

## 💡 Use Cases

### 🤖 Conversational AI Agents
```python
class ChatAgent:
    def __init__(self):
        self.memory = MemoryManager()
    
    def process(self, user_input):
        # Store conversation
        self.memory.add(
            user_input,
            memory_type="short_term",
            session_id=self.session_id
        )
        
        # Retrieve context
        context = self.memory.retrieve(user_input, limit=5)
        
        # Generate response with context
        response = self.generate_response(user_input, context)
        return response
```

### 📚 Knowledge Management
```python
# Store facts with concepts
memory.add(
    "Python was created in 1991 by Guido van Rossum",
    memory_type="semantic",
    concepts=["python", "history", "programming"],
    importance=0.8
)

# Query by concepts
python_facts = memory.retrieve(concepts=["python"], limit=10)
```

### 🔧 Skill Learning
```python
# Store procedures
memory.add(
    {"procedure": "Deploy to AWS", "steps": [...]},
    memory_type="procedural",
    procedure_name="aws_deployment",
    skill_level=0.7
)

# Track execution success
memory.update_execution(procedure_id, success=True)
```

## 🏗️ Architecture

```
AgentMemory/
├── core/
│   ├── memory_manager.py    # Central orchestrator
│   ├── memory_types.py      # Memory type implementations
│   └── memory_entry.py      # Memory data structures
├── stores/
│   ├── base.py             # Abstract store interface
│   └── vector.py           # Vector similarity stores
├── retrievers/
│   ├── base.py             # Abstract retriever interface
│   └── semantic.py         # Semantic search implementation
└── utils/
    └── embeddings.py       # Embedding utilities
```

## 🔬 Advanced Features

### Memory Consolidation
```python
# Automatic consolidation from short-term to long-term
manager = MemoryManager(consolidation_threshold=10)

# Manual consolidation
stats = manager.consolidate_memories()
print(f"Promoted to long-term: {stats['promoted_to_long_term']}")
```

### Custom Metadata
```python
memory.add(
    "Important event",
    memory_type="episodic",
    custom_metadata={
        "location": "San Francisco",
        "participants": ["Alice", "Bob"],
        "outcome": "successful"
    }
)
```

### Vector Search with Embeddings
```python
from agentmemory.utils.embeddings import EmbeddingManager

embedder = EmbeddingManager()
embedding = embedder.encode_single("Machine learning concept")

memory.add(
    "Neural networks are inspired by biological neurons",
    memory_type="semantic",
    embedding=embedding
)
```

## 🤝 Integration with Popular Frameworks

### LangChain
```python
from langchain.memory import ConversationBufferMemory
from agentmemory import MemoryManager

class AgentMemoryWrapper(ConversationBufferMemory):
    def __init__(self):
        super().__init__()
        self.agent_memory = MemoryManager()
    
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        self.agent_memory.add(
            {"input": inputs, "output": outputs},
            memory_type="episodic"
        )
```

### AutoGen
```python
from autogen import AssistantAgent
from agentmemory import MemoryManager

class MemoryAgent(AssistantAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.memory = MemoryManager()
    
    def receive(self, message, sender):
        # Store message in memory
        self.memory.add(
            message,
            memory_type="short_term",
            agent_id=sender.name
        )
        return super().receive(message, sender)
```

## 📊 Performance Considerations

- **Embedding Dimension**: Default 384 (all-MiniLM-L6-v2), adjustable for performance
- **Capacity Limits**: Configurable per memory type to manage resource usage
- **Batch Operations**: Use batch methods for bulk operations
- **Vector Store Choice**: NumPy for small-scale, FAISS for production

## 🛠️ Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black agentmemory/
ruff check agentmemory/
```

Type checking:
```bash
mypy agentmemory/
```

## 📈 Benchmarks

| Operation | 1K Memories | 10K Memories | 100K Memories |
|-----------|-------------|--------------|---------------|
| Add       | 0.8ms       | 0.9ms        | 1.1ms         |
| Retrieve  | 2.3ms       | 8.7ms        | 45ms          |
| Search    | 3.1ms       | 12ms         | 89ms          |

*Tested on MacBook Pro M1, 16GB RAM*

## 🗺️ Roadmap

- [ ] Distributed memory stores (Redis, PostgreSQL)
- [ ] Graph-based memory relationships
- [ ] Memory compression and summarization
- [ ] Multi-agent memory sharing
- [ ] Memory versioning and rollback
- [ ] Advanced consolidation strategies
- [ ] Memory attention mechanisms
- [ ] Integration with more frameworks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hanish Keloth**
- GitHub: [@hanishkeloth](https://github.com/hanishkeloth)
- Email: hanishkeloth1256@gmail.com

## 🙏 Acknowledgments

- Inspired by cognitive architectures and human memory systems
- Built to address gaps identified in current agentic AI frameworks
- Thanks to the open-source AI community for continuous innovation

## 📚 Citations

If you use AgentMemory in your research or projects, please cite:

```bibtex
@software{agentmemory2025,
  author = {Keloth, Hanish},
  title = {AgentMemory: Advanced Memory Management for AI Agents},
  year = {2025},
  url = {https://github.com/hanishkeloth/agentmemory}
}
```

## 🔗 Links

- [Documentation](https://github.com/hanishkeloth/agentmemory/wiki)
- [Issues](https://github.com/hanishkeloth/agentmemory/issues)
- [Discussions](https://github.com/hanishkeloth/agentmemory/discussions)
- [PyPI Package](https://pypi.org/project/agentmemory/)

---

**Note**: This project is in beta. APIs may change in future versions. Please report any issues or suggestions!