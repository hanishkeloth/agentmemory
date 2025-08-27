# AgentMemory 🧠

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/hanishkeloth/agentmemory)

**AgentMemory** ist ein fortschrittliches Speicherverwaltungs-Framework für KI-Agenten, das persistente, hierarchische und semantische Speicherfähigkeiten bietet. Es behebt die kritische Lücke in der aktuellen agentischen KI-Entwicklung: robustes Speicher- und Kontextmanagement über Agentensitzungen hinweg.

## 🚀 Hauptmerkmale

### 🎯 Mehrere Speichertypen
- **Kurzzeitgedächtnis**: FIFO-Puffer mit TTL für unmittelbaren Kontext
- **Langzeitgedächtnis**: Wichtigkeitsbasierte Aufbewahrung für wertvolle Informationen
- **Episodisches Gedächtnis**: Zeitliche Organisation von Ereignissequenzen
- **Semantisches Gedächtnis**: Konzeptbasiertes Wissen mit Beziehungen
- **Prozedurales Gedächtnis**: Fähigkeiten und Anleitungswissen mit Ausführungsverfolgung

### 🔍 Erweiterte Abfrage
- **Semantische Suche**: Vektor-Ähnlichkeitssuche mit Einbettungen
- **Gefilterte Abfragen**: Abruf nach Typ, Tags, Wichtigkeit oder benutzerdefinierten Metadaten
- **Beziehungsnavigation**: Verfolgung von Speicherassoziationen und -beziehungen
- **Zeit-Zerfalls-Bewertung**: Automatische Relevanzberechnung mit Zerfall

### 💾 Persistenz & Speicherung
- **Mehrere Backends**: In-Memory, Vektorspeicher (NumPy/FAISS)
- **Speichern/Laden**: JSON-Serialisierung für Speicherpersistenz
- **Batch-Operationen**: Effiziente Massen-Hinzufüge-/Löschoperationen

### 🔄 Speicherverwaltung
- **Auto-Konsolidierung**: Automatische Beförderung von Kurzzeit- zu Langzeitgedächtnis
- **Wichtigkeitsbewertung**: Konfigurierbare Wichtigkeitsschwellen
- **Speicherassoziationen**: Beziehungen zwischen Erinnerungen erstellen
- **Statistikverfolgung**: Überwachung von Speichernutzung und -mustern

## 📦 Installation

```bash
pip install agentmemory
```

Für die Entwicklung:
```bash
git clone https://github.com/hanishkeloth/agentmemory.git
cd agentmemory
pip install -e ".[dev]"
```

## 🎓 Schnellstart

```python
from agentmemory import MemoryManager

# Speichermanager initialisieren
memory = MemoryManager()

# Erinnerungen hinzufügen
memory.add(
    "Benutzer bevorzugt Python für Data Science",
    memory_type="long_term",
    importance=0.9,
    tags=["preference", "python"]
)

# Relevante Erinnerungen abrufen
memories = memory.retrieve(
    query="Welche Programmiersprache verwenden?",
    limit=5
)

# Assoziationen erstellen
memory.create_association(memory1_id, memory2_id, "related_to")

# Für Persistenz speichern
memory.save("agent_memories.json")
```

## 💡 Anwendungsfälle

### 🤖 Konversations-KI-Agenten
```python
class ChatAgent:
    def __init__(self):
        self.memory = MemoryManager()
    
    def process(self, user_input):
        # Konversation speichern
        self.memory.add(
            user_input,
            memory_type="short_term",
            session_id=self.session_id
        )
        
        # Kontext abrufen
        context = self.memory.retrieve(user_input, limit=5)
        
        # Antwort mit Kontext generieren
        response = self.generate_response(user_input, context)
        return response
```

### 📚 Wissensmanagement
```python
# Fakten mit Konzepten speichern
memory.add(
    "Python wurde 1991 von Guido van Rossum erstellt",
    memory_type="semantic",
    concepts=["python", "geschichte", "programmierung"],
    importance=0.8
)

# Abfrage nach Konzepten
python_facts = memory.retrieve(concepts=["python"], limit=10)
```

### 🔧 Fertigkeitserlernung
```python
# Prozeduren speichern
memory.add(
    {"procedure": "AWS-Bereitstellung", "steps": [...]},
    memory_type="procedural",
    procedure_name="aws_deployment",
    skill_level=0.7
)

# Ausführungserfolg verfolgen
memory.update_execution(procedure_id, success=True)
```

## 🏗️ Architektur

```
AgentMemory/
├── core/
│   ├── memory_manager.py    # Zentraler Orchestrator
│   ├── memory_types.py      # Speichertyp-Implementierungen
│   └── memory_entry.py      # Speicher-Datenstrukturen
├── stores/
│   ├── base.py             # Abstrakte Store-Schnittstelle
│   └── vector.py           # Vektor-Ähnlichkeitsspeicher
├── retrievers/
│   ├── base.py             # Abstrakte Retriever-Schnittstelle
│   └── semantic.py         # Semantische Suchimplementierung
└── utils/
    └── embeddings.py       # Einbettungs-Utilities
```

## 🔬 Erweiterte Funktionen

### Speicherkonsolidierung
```python
# Automatische Konsolidierung von Kurzzeit- zu Langzeitgedächtnis
manager = MemoryManager(consolidation_threshold=10)

# Manuelle Konsolidierung
stats = manager.consolidate_memories()
print(f"Zu Langzeitgedächtnis befördert: {stats['promoted_to_long_term']}")
```

### Benutzerdefinierte Metadaten
```python
memory.add(
    "Wichtiges Ereignis",
    memory_type="episodic",
    custom_metadata={
        "location": "San Francisco",
        "participants": ["Alice", "Bob"],
        "outcome": "erfolgreich"
    }
)
```

### Vektorsuche mit Einbettungen
```python
from agentmemory.utils.embeddings import EmbeddingManager

embedder = EmbeddingManager()
embedding = embedder.encode_single("Machine Learning Konzept")

memory.add(
    "Neuronale Netze sind von biologischen Neuronen inspiriert",
    memory_type="semantic",
    embedding=embedding
)
```

## 🤝 Integration mit beliebten Frameworks

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
        # Nachricht im Speicher speichern
        self.memory.add(
            message,
            memory_type="short_term",
            agent_id=sender.name
        )
        return super().receive(message, sender)
```

## 📊 Leistungsüberlegungen

- **Einbettungsdimension**: Standard 384 (all-MiniLM-L6-v2), anpassbar für Leistung
- **Kapazitätsgrenzen**: Konfigurierbar pro Speichertyp zur Verwaltung der Ressourcennutzung
- **Batch-Operationen**: Verwenden Sie Batch-Methoden für Massenoperationen
- **Vektorspeicher-Wahl**: NumPy für kleine Maßstäbe, FAISS für Produktion

## 🛠️ Entwicklung

Tests ausführen:
```bash
pytest tests/
```

Code formatieren:
```bash
black agentmemory/
ruff check agentmemory/
```

Typprüfung:
```bash
mypy agentmemory/
```

## 📈 Benchmarks

| Operation | 1K Speicher | 10K Speicher | 100K Speicher |
|-----------|-------------|--------------|---------------|
| Hinzufügen | 0.8ms      | 0.9ms        | 1.1ms         |
| Abrufen   | 2.3ms       | 8.7ms        | 45ms          |
| Suchen    | 3.1ms       | 12ms         | 89ms          |

*Getestet auf MacBook Pro M1, 16GB RAM*

## 🗺️ Roadmap

- [ ] Verteilte Speicherspeicher (Redis, PostgreSQL)
- [ ] Graphbasierte Speicherbeziehungen
- [ ] Speicherkomprimierung und Zusammenfassung
- [ ] Multi-Agenten-Speicherfreigabe
- [ ] Speicherversionierung und Rollback
- [ ] Erweiterte Konsolidierungsstrategien
- [ ] Speicher-Aufmerksamkeitsmechanismen
- [ ] Integration mit mehr Frameworks

## 🤝 Beitragen

Beiträge sind willkommen! Bitte zögern Sie nicht, einen Pull Request einzureichen.

1. Repository forken
2. Feature-Branch erstellen (`git checkout -b feature/amazing-feature`)
3. Änderungen committen (`git commit -m 'Add amazing feature'`)
4. Zum Branch pushen (`git push origin feature/amazing-feature`)
5. Pull Request öffnen

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## 👨‍💻 Autor

**Hanish Keloth**
- GitHub: [@hanishkeloth](https://github.com/hanishkeloth)
- Email: hanishkeloth@gmail.com

## 🙏 Danksagungen

- Inspiriert von kognitiven Architekturen und menschlichen Gedächtnissystemen
- Entwickelt zur Behebung von Lücken in aktuellen agentischen KI-Frameworks
- Dank an die Open-Source-KI-Community für kontinuierliche Innovation

## 📚 Zitierung

Wenn Sie AgentMemory in Ihrer Forschung oder Ihren Projekten verwenden, zitieren Sie bitte:

```bibtex
@software{agentmemory2025,
  author = {Keloth, Hanish},
  title = {AgentMemory: Erweiterte Speicherverwaltung für KI-Agenten},
  year = {2025},
  url = {https://github.com/hanishkeloth/agentmemory}
}
```

## 🔗 Links

- [Dokumentation](https://github.com/hanishkeloth/agentmemory/wiki)
- [Issues](https://github.com/hanishkeloth/agentmemory/issues)
- [Diskussionen](https://github.com/hanishkeloth/agentmemory/discussions)
- [PyPI-Paket](https://pypi.org/project/agentmemory/)

---

**Hinweis**: Dieses Projekt befindet sich in der Beta-Phase. APIs können sich in zukünftigen Versionen ändern. Bitte melden Sie Probleme oder Vorschläge!