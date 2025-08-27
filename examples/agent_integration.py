"""Example of integrating AgentMemory with an AI agent."""

from typing import Dict, List, Any
from agentmemory import MemoryManager


class IntelligentAgent:
    """Example AI agent with memory capabilities."""
    
    def __init__(self, agent_id: str = "agent_001"):
        """Initialize agent with memory system."""
        self.agent_id = agent_id
        self.memory_manager = MemoryManager(
            short_term_capacity=20,
            long_term_capacity=1000,
            consolidation_threshold=10
        )
        self.current_session = None
        self.context_window = []
    
    def start_session(self, session_id: str):
        """Start a new conversation session."""
        self.current_session = session_id
        self.context_window = []
        print(f"🚀 Session started: {session_id}")
    
    def process_input(self, user_input: str) -> str:
        """Process user input with memory context."""
        # Store input in short-term memory
        input_memory = self.memory_manager.add(
            content=f"User: {user_input}",
            memory_type="short_term",
            agent_id=self.agent_id,
            session_id=self.current_session,
            tags=self._extract_tags(user_input),
            importance=self._calculate_importance(user_input)
        )
        
        # Retrieve relevant context
        context = self._retrieve_context(user_input)
        
        # Generate response (mock)
        response = self._generate_response(user_input, context)
        
        # Store response in memory
        response_memory = self.memory_manager.add(
            content=f"Agent: {response}",
            memory_type="short_term",
            agent_id=self.agent_id,
            session_id=self.current_session,
            tags=["response"],
            importance=0.5
        )
        
        # Create association between input and response
        self.memory_manager.create_association(
            input_memory.id,
            response_memory.id,
            relation_type="response_to"
        )
        
        # Update context window
        self.context_window.append((user_input, response))
        if len(self.context_window) > 5:
            self.context_window.pop(0)
        
        return response
    
    def learn_fact(self, fact: str, concepts: List[str]):
        """Learn a new fact and store in semantic memory."""
        memory = self.memory_manager.add(
            content=fact,
            memory_type="semantic",
            concepts=concepts,
            importance=0.8,
            agent_id=self.agent_id
        )
        print(f"📚 Learned: {fact[:50]}...")
        return memory
    
    def learn_procedure(self, name: str, steps: List[str]):
        """Learn a new procedure."""
        memory = self.memory_manager.add(
            content={
                "procedure": name,
                "steps": steps
            },
            memory_type="procedural",
            procedure_name=name,
            steps=steps,
            skill_level=0.5,
            importance=0.7,
            agent_id=self.agent_id
        )
        print(f"🔧 Learned procedure: {name}")
        return memory
    
    def recall_facts(self, concepts: List[str]) -> List[str]:
        """Recall facts related to concepts."""
        memories = self.memory_manager.retrieve(
            memory_types=["semantic"],
            concepts=concepts,
            limit=5
        )
        return [m.content for m in memories]
    
    def end_session(self):
        """End current session and consolidate memories."""
        if self.current_session:
            # Mark session as episode
            episode_memories = self.memory_manager.retrieve(
                memory_types=["short_term"],
                limit=100
            )
            
            # Convert important short-term to episodic
            for memory in episode_memories:
                if memory.metadata.session_id == self.current_session:
                    if memory.metadata.importance_score >= 0.5:
                        self.memory_manager.add(
                            content=memory.content,
                            memory_type="episodic",
                            episode_id=self.current_session,
                            importance=memory.metadata.importance_score,
                            tags=memory.metadata.tags,
                            agent_id=self.agent_id
                        )
            
            # Consolidate memories
            stats = self.memory_manager.consolidate_memories()
            print(f"💾 Session ended. Consolidation: {stats}")
            
            self.current_session = None
            self.context_window = []
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text (simplified)."""
        keywords = []
        important_words = ["python", "code", "help", "error", "question", "problem"]
        for word in important_words:
            if word.lower() in text.lower():
                keywords.append(word)
        return keywords
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score (simplified)."""
        if "important" in text.lower() or "remember" in text.lower():
            return 0.9
        elif "?" in text:
            return 0.7
        elif len(text) > 100:
            return 0.6
        else:
            return 0.4
    
    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context for query."""
        # Get recent conversation context
        recent = self.memory_manager.retrieve(
            memory_types=["short_term"],
            limit=3
        )
        
        # Get relevant long-term memories
        relevant = self.memory_manager.retrieve(
            query=query,
            memory_types=["long_term", "semantic"],
            limit=2
        )
        
        context = []
        for memory in recent + relevant:
            context.append({
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.metadata.importance_score
            })
        
        return context
    
    def _generate_response(self, input_text: str, context: List[Dict[str, Any]]) -> str:
        """Generate response based on input and context (mock)."""
        # This is a simplified mock response generator
        if "?" in input_text:
            if context:
                return f"Based on my memory, I recall: {context[0]['content'][:100]}..."
            return "I'll help you with that. Let me think..."
        elif "remember" in input_text.lower():
            return "I've stored that information for future reference."
        else:
            return "I understand. Please tell me more."
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return self.memory_manager.get_statistics()


def main():
    """Demonstrate agent with memory integration."""
    print("🤖 AgentMemory Integration Example\n")
    
    # Create agent
    agent = IntelligentAgent("assistant_001")
    
    # Learn some facts
    print("1. Teaching agent some facts:")
    agent.learn_fact(
        "Python was created by Guido van Rossum and first released in 1991",
        ["python", "history", "programming"]
    )
    agent.learn_fact(
        "Machine learning is a subset of artificial intelligence",
        ["machine_learning", "ai", "technology"]
    )
    
    # Learn a procedure
    print("\n2. Teaching agent a procedure:")
    agent.learn_procedure(
        "debug_python_error",
        [
            "Read the error traceback",
            "Identify the line causing the error",
            "Check variable types and values",
            "Add print statements or use debugger",
            "Fix the issue and test"
        ]
    )
    
    # Start conversation session
    print("\n3. Starting conversation session:")
    agent.start_session("session_001")
    
    # Simulate conversation
    interactions = [
        "What is Python?",
        "Remember that I prefer using VS Code for development",
        "Can you help me debug an error?",
        "What facts do you know about machine learning?"
    ]
    
    for user_input in interactions:
        print(f"\n👤 User: {user_input}")
        response = agent.process_input(user_input)
        print(f"🤖 Agent: {response}")
    
    # Check memory stats
    print("\n4. Memory statistics during session:")
    stats = agent.get_memory_stats()
    print(f"   Total memories: {stats['total_memories']}")
    for mem_type, count in stats["by_type"].items():
        print(f"   - {mem_type}: {count}")
    
    # End session
    print("\n5. Ending session and consolidating:")
    agent.end_session()
    
    # Final stats
    print("\n6. Final memory statistics:")
    final_stats = agent.get_memory_stats()
    print(f"   Total memories: {final_stats['total_memories']}")
    for mem_type, count in final_stats["by_type"].items():
        print(f"   - {mem_type}: {count}")
    
    # Recall facts
    print("\n7. Recalling facts about Python:")
    python_facts = agent.recall_facts(["python"])
    for fact in python_facts:
        print(f"   - {fact[:80]}...")
    
    print("\n✅ Agent integration example completed!")


if __name__ == "__main__":
    main()