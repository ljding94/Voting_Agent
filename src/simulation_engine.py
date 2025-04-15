import json
import random
import os
from agent import Agent, Community


class SimulationEngine:
    def __init__(self, api_key, agents_file, topic, rounds, p=0.2):
        # Initialize agents with the memory config
        self.agents = Agent.load_agents_from_file(api_key, agents_file)
        self.topic = topic
        self.rounds = rounds

        # Initialize community using small-world network parameters from config
        self.community = Community(self.agents, p)

        self.voting_results = []
        self.sentiment_scores = []
        self.current_round = 0

    def load_simulation_config(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def run_simulation(self, folder, finfo):
        log_messages = []

        def log(msg):
            print(msg)
            log_messages.append(msg)

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
        with open(log_file_path, 'w') as log_file:
            # Write all log messages to the file
            log_file.write("\n".join(log_messages))

    def collect_votes(self):
        """Collects votes from all agents based on their sentiment response"""
        round_votes = {}
        round_sentiments = {}

        # Create clear voting prompt
        vote_prompt = f"Based on the discussion so far about '{self.topic}', what is your position? Express your sentiment from -5 (strongly against) to +5 (strongly support)."

        print("\n--- Voting ---")
        print(vote_prompt)
        print("\n----------------------------------------\n")

        for agent in self.agents:
            # Get response with sentiment value
            vote_response = agent.generate_response(vote_prompt)

            # Print the agent's voting response
            print(f"{agent.name}'s vote:")
            print(f"Reasoning: {vote_response.reasoning}")
            print(f"Answer: {vote_response.answer}")
            print(f"Sentiment: {vote_response.sentiment}")
            print("----------------------------------------\n")

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

        print("Round Sentiments:", round_sentiments)
        print("Round Votes:", round_votes)
        print("\n----------------------------------------\n")
        return round_votes, round_sentiments

    def save_sentiment_scores(self, filepath):
        """Save sentiment scores to a JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.sentiment_scores, f, indent=2)
        print(f"Sentiment scores saved to {filepath}")


if __name__ == "__main__":
    engine = SimulationEngine("config/agents.json", "config/simulation_config.json")
    engine.run_simulation("data/voting_results.json")
