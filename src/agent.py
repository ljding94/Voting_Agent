# Defines the agent class with persona and LLM integration
import json
from datetime import datetime
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from typing import Optional
from pydantic import BaseModel, Field
import networkx as nx
import random


class AgentResponse(BaseModel):
    reflection: str = Field(description="A reflection on who you are and your background")
    reasoning: str = Field(description="The reasoning behind the response")
    answer: str = Field(description="The main answer of the agent")
    sentiment: int = Field(description="The sentiment of the answer towards the question, from -5 to 5")


class Agent:
    def __init__(self, api_key, persona, memory_config=None):
        self.id = persona["id"]
        self.name = persona["name"]
        self.background = persona.get("background", "")
        self.personality = persona.get("personality", "")  # Optional personality trait
        self.upbringing_category = persona.get("upbringing_category", None)  # Optional upbringing category

        # Enhanced memory system with categories
        self.memory = {
            "personal_statements": [],  # Agent's own responses
            "observed_statements": [],  # Other agents' statements
        }

        # Replace OpenAI with ChatOpenAI
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", api_key=api_key)

    def generate_response(self, prompt, context=""):
        """
        Generates a structured response based on the given prompt and optional context.
        Returns a formatted AgentResponse object with reasoning, answer, and sentiment.
        """
        # Build context from memory relevant to the prompt
        memory_context = self._get_memory_context()

        # Create system message with agent's background info
        system_content = f"You are {self.name}.\n"
        if self.background:
            system_content += f"Background: {self.background}\n"
            system_content += f"Personality: {self.personality}\n"

        # Setup Pydantic parser for structured output
        parser = PydanticOutputParser(pydantic_object=AgentResponse)

        # User content - include the format instructions for structured output
        user_content = ""
        user_content += f"You must respond in the following format:\n{parser.get_format_instructions()}\n\n"

        if memory_context:
            user_content += f"What you remember: {memory_context}\n\n"

        if context:
            user_content += f"Context: {context}\n\n"

        user_content += f"Question: {prompt}"

        # Create message list
        messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]

        # Get response
        try:
            raw_response = self.llm.invoke(messages).content
            agent_response = parser.parse(raw_response)

            # Store full response text in memory
            formatted_response = f"Reasoning: {agent_response.reasoning}\nAnswer: {agent_response.answer}\nSentiment: {agent_response.sentiment}"
            self.record_own_statement(prompt, formatted_response)

            response = agent_response

        except Exception as e:
            # Fallback for parsing errors
            print(f"Error parsing structured response: {e}")
            fallback_response = AgentResponse(
                reasoning="Sorry, I had trouble structuring my response.", answer=raw_response if "raw_response" in locals() else "Error generating response.", sentiment=0
            )
            self.record_own_statement(prompt, f"Error: {e}\nFallback: {fallback_response.answer}")
            response = fallback_response

        self._clean_memory()
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
            "speaker": speaker_name,
            "statement": statement,
        }
        self.memory["observed_statements"].append(observation)

    def _get_memory_context(self):
        memories = []
        if self.memory["personal_statements"]:
            memories.append(f"yourself said: {self.memory['personal_statements'][-1]['response']}")
        # Check for observed statements
        for statement in reversed(self.memory["observed_statements"]):
            # Get the statement content, handling different types
            statement_text = statement["statement"]

            # If it's an AgentResponse object, convert to string
            if hasattr(statement_text, "answer"):
                statement_text = f"Answer: {statement_text.answer}\nReasoning: {statement_text.reasoning}\nSentiment: {statement_text.sentiment}"

            memories.append(f"{statement['speaker']} said: {statement_text}")

        return "\n".join(memories) if memories else ""

    def _clean_memory(self):
        """Clean up memory to avoid overflow"""
        self.memory = {
            "personal_statements": [],  # Agent's own responses
            "observed_statements": [],  # Other agents' statements
        }

    @staticmethod
    def load_agents_from_file(api_key, filepath, memory_config=None):
        """
        Loads agent personas from a single JSON file and returns a list of Agent instances.
        """
        with open(filepath, "r") as file, open(filepath, "r") as file:
            data = json.load(file)
        agents = [Agent(api_key, persona, memory_config) for persona in data.get("agents", [])]
        return agents


class Community:
    def __init__(self, agents, p=0.5):
        """
        Initializes a community of agents and sets up their connectivity using
        a random network model.

        Parameters:
        - agents: list of Agent instances.
        - p: Probability for edge creation between any two agents.
        """
        self.agents = agents
        self.connectivity = self._build_random_network(len(agents), p)

    def _build_random_network(self, num_agents, p):
        """
        Builds a random network using the Erdős–Rényi model and returns a
        mapping of agent indices to a list of neighbor indices.
        """
        graph = nx.erdos_renyi_graph(num_agents, p)
        connectivity = {node: list(graph.neighbors(node)) for node in graph.nodes()}
        return connectivity

    def discussion_round(self, topic):
        """
        Simulate a discussion round on a given topic.
        Each agent generates a response and shares it with its neighbors.

        Parameters:
        - topic: The topic for discussion.
        - communication_probability: Probability that a neighbor will listen to a statement.
        """
        for idx, agent in enumerate(self.agents):
            # Generate response using agent's generate_response method
            response = agent.generate_response(topic)
            print(f"{agent.name} says: {response.answer}")

            # Retrieve neighbor indices for the current agent
            neighbors = self.connectivity.get(idx, [])
            print(f"  → {agent.name}'s neighbors: {[self.agents[n].name for n in neighbors]}")
            for neighbor_idx in neighbors:
                # Optionally, decide probabilistically if the neighbor hears the response
                # Record that the neighbor observed this statement
                self.agents[neighbor_idx].record_observed_statement(agent.name, response.answer)
                print(f"  → {self.agents[neighbor_idx].name} heard this message")