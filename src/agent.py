# Defines the agent class with persona and LLM integration
import json
from datetime import datetime
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class Agent:
    def __init__(self, persona, memory_config=None):
        self.id = persona["id"]
        self.name = persona["name"]
        self.role = persona.get("role", "Agent")
        self.background = persona.get("background", "")
        self.personality = persona.get("personality", {})
        self.knowledge = persona.get("knowledge", {})
        self.relationships = persona.get("relationships", {}) # probability weight I will listen to this person

        # Store preferences (new field in agents.json)
        self.preferences = persona.get("preferences", {})

        # Memory configuration with defaults
        self.memory_config = memory_config or {"max_relevant_items": 3}

        # Enhanced memory system with categories
        self.memory = {
            "personal_statements": [],  # Agent's own responses
            "observed_statements": [],  # Other agents' statements
            "voting_history": [],  # Records of votes cast (own and others)
            "discussion_topics": [],  # Topics discussed
        }

        # Replace OpenAI with ChatOpenAI
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    def generate_response(self, prompt, context=""):
        """
        Generates a response based on the given prompt and optional context.
        It constructs a detailed prompt incorporating the agent's persona and memory.
        """
        # Build context from memory relevant to the prompt
        memory_context = self._get_relevant_memory_context(prompt)

        # Create system message with agent's background info
        system_content = f"You are Agent {self.name} ({self.role}).\n"

        # Include background if available
        if self.background:
            system_content += f"Background: {self.background}\n"

        # Include personality if available
        if self.personality:
            system_content += f"Personality: {self.personality}\n"

        # Add preferences (new addition)
        if self.preferences:
            preferences_text = ", ".join([f"{k}: {v}" for k, v in self.preferences.items()])
            system_content += f"Preferences: {preferences_text}\n"

        system_content += "Respond as this character would, maintaining their personality and background."

        # Create user message with context, memory and prompt
        user_content = ""
        if memory_context:
            user_content += f"What you remember: {memory_context}\n\n"

        if context:
            user_content += f"Context: {context}\n\n"

        user_content += f"Question: {prompt}"

        # Create message list
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]

        # Get response
        response = self.llm.invoke(messages).content

        # Store own response in memory
        self.record_own_statement(prompt, response)

        return response

    def record_own_statement(self, prompt, response):
        """Record agent's own statement with metadata"""
        statement = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
        }
        self.memory["personal_statements"].append(statement)

    def record_observed_statement(self, speaker_name, statement):
        """Record another agent's statement"""
        observation = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker_name,
            "statement": statement,
        }
        self.memory["observed_statements"].append(observation)

    def record_vote(self, round_number, own_vote, all_votes=None):
        """Record voting results"""
        vote_record = {"timestamp": datetime.now().isoformat(), "round": round_number, "own_vote": own_vote, "all_votes": all_votes or {}}
        self.memory["voting_history"].append(vote_record)

    def record_discussion_topic(self, topic):
        """Record a discussion topic"""
        topic_entry = {"timestamp": datetime.now().isoformat(), "topic": topic}
        self.memory["discussion_topics"].append(topic_entry)

    def _get_relevant_memory_context(self, prompt):
        """Retrieve memories relevant to the current prompt"""
        max_items = self.memory_config.get("max_relevant_items", 3)

        # Simple keyword-based relevance (could be enhanced with embeddings)
        relevant_memories = []

        # Check for relevant observed statements
        for statement in reversed(self.memory["observed_statements"]):
            if any(word in statement["statement"].lower() for word in prompt.lower().split()):
                relevant_memories.append(f"{statement['speaker']} said: {statement['statement']}")
            if len(relevant_memories) >= max_items:
                break

        # Check voting history
        if "vote" in prompt.lower():
            for vote in reversed(self.memory["voting_history"]):
                relevant_memories.append(f"In round {vote['round']}, you voted: {vote['own_vote']}")
                break

        return "\n".join(relevant_memories) if relevant_memories else ""

    def update_memory(self, event):
        """
        Legacy method for backward compatibility
        Adds an event to personal statements
        """
        self.record_own_statement("Legacy event", event)
        self.memory.append(event)  # Keep the old list for compatibility

    @staticmethod
    def load_agents_from_file(filepath, memory_config=None):
        """
        Loads agent personas from a single JSON file and returns a list of Agent instances.
        """
        with open(filepath, "r") as file:
            data = json.load(file)
        agents = [Agent(persona, memory_config) for persona in data.get("agents", [])]
        return agents
