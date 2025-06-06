import json
import random
import os
from agent import Agent, Community

log_messages = []


def log(msg):
    print(msg)
    log_messages.append(msg)


class SimulationEngine:
    def __init__(self, api_key, nIA, nID, nAA, nAD, topic, rounds, p=0.2):
        # Initialize agents with the memory config
        # self.agents = Agent.load_agents_from_file(api_key, agents_file)
        self.agents = Agent.generate_agents_by_type(api_key, {"IA": nIA, "ID": nID, "AA": nAA, "AD": nAD})
        self.topic = topic
        self.rounds = rounds

        # Initialize community using small-world network parameters from config
        self.community = Community(self.agents, p)

        self.voting_results = []
        self.sentiment_scores = []
        self.current_round = 0

    def load_simulation_config(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def run_simulation(self, folder, finfo):

        log(f"Starting simulation on topic: {self.topic}")

        # Initial voting round (round 0) to establish starting positions
        log("\n--- Initial Voting ---")
        initial_prompt = f"Based on your background and personality, what is your initial stance on: {self.topic}? Just explain briefly."
        for agent in self.agents:
            response = agent.generate_response(initial_prompt)
            log(f"{agent.name}'s initial stance:")
            log(f"Answer: {response.answer}")
            log(f"Reasoning: {response.reasoning}")
            log("\n----------------------------------------\n")

        # Fix: Unpack both values from collect_votes()
        initial_votes, initial_sentiments = self.collect_votes()
        self.voting_results.append(initial_votes)
        self.sentiment_scores.append(initial_sentiments)
        self.current_round = 0

        # Main simulation rounds
        for i in range(self.rounds):
            self.current_round = i + 1
            log(f"\n--- Round {self.current_round} ---")
            # Use the Community's discussion round instead of the SimulationEngine's method
            self.community.discussion_round(self.topic)
            round_votes, round_sentiments = self.collect_votes()
            self.voting_results.append(round_votes)
            self.sentiment_scores.append(round_sentiments)

        log(f"self.sentiment_scores: {self.sentiment_scores}")

        # Save the detailed voting results to the specified file path
        # Save sentiment scores
        self.save_sentiment_scores(f"{folder}/{finfo}_sentiment_scores.json")
        log_file_path = os.path.join(folder, f"{finfo}_log.txt")
        with open(log_file_path, "w") as log_file:
            # Write all log messages to the file
            log_file.write("\n".join(str(msg) for msg in log_messages))

    def collect_votes(self):
        """Collects votes from all agents based on their sentiment response"""
        round_votes = {}
        round_sentiments = {}

        # Create clear voting prompt
        vote_prompt = f"Based on the discussion so far about '{self.topic}', what is your position? Express your sentiment from -5 (strongly against) to +5 (strongly support)."

        log("\n--- Voting ---")
        log(vote_prompt)
        log("\n----------------------------------------\n")

        for agent in self.agents:
            # Get response with sentiment value
            vote_response = agent.generate_response(vote_prompt)

            # Print the agent's voting response
            log(f"{agent.name}'s vote:")
            log(f"Reasoning: {vote_response.reasoning}")
            log(f"Answer: {vote_response.answer}")
            log(f"Sentiment: {vote_response.sentiment}")
            log("----------------------------------------\n")

            # Determine vote based on sentiment value
            sentiment = vote_response.sentiment

            # Map sentiment to vote categories
            if sentiment > 0:
                vote = "Support"
            elif sentiment < 0:
                vote = "Against"
            else:
                vote = "Abstain"

            round_votes[agent.name] = vote
            round_sentiments[agent.name] = sentiment

            # Record the voting response in agent's memory
            formatted_response = f"Reasoning: {vote_response.reasoning}\nAnswer: {vote_response.answer}\nSentiment: {sentiment}"
            agent.record_own_statement(vote_prompt, formatted_response)

        log("Round Sentiments:")
        log(round_sentiments)
        log("Round Votes:")
        log(round_votes)
        log("\n----------------------------------------\n")
        return round_votes, round_sentiments

    def save_sentiment_scores(self, filepath):
        """Save sentiment scores to a JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Transform sentiment scores to include agent tags
        formatted_sentiment_scores = []
        for round_idx, round_sentiments in enumerate(self.sentiment_scores):
            formatted_round = {}
            for agent_name, sentiment in round_sentiments.items():
                # Find the agent to get their tag
                agent = next(agent for agent in self.agents if agent.name == agent_name)
                formatted_round[f"{agent.tag}_{agent_name}"] = sentiment
            formatted_sentiment_scores.append(formatted_round)

        with open(filepath, "w") as f:
            json.dump(formatted_sentiment_scores, f, indent=2)
        log(f"Sentiment scores saved to {filepath}")
