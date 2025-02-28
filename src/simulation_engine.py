import json
import random
from agent import Agent


class SimulationEngine:
    def __init__(self, agents_file, simulation_config_file):
        self.simulation_config = self.load_simulation_config(simulation_config_file)

        # Extract memory configuration
        memory_config = self.simulation_config.get("agent_memory", {})

        # Initialize agents with the memory config
        self.agents = Agent.load_agents_from_file(agents_file, memory_config)
        self.voting_results = []
        self.current_round = 0

    def load_simulation_config(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def run_simulation(self, results_filepath="data/voting_results.json"):
        rounds = self.simulation_config.get("rounds", 1)
        discussion_topic = self.simulation_config.get("discussion_topic", "No topic defined")
        print("Starting simulation on topic:", discussion_topic)

        # Record the discussion topic for each agent
        for agent in self.agents:
            agent.record_discussion_topic(discussion_topic)

        # Initial voting round (round 0) to establish starting positions
        print("\n--- Initial Voting ---")
        initial_prompt = f"Based on your background and personality, what is your initial stance on: {discussion_topic}? Just explain briefly."
        for agent in self.agents:
            response = agent.generate_response(initial_prompt)
            print(f"{agent.name}'s initial stance: {response}")
            print("\n----------------------------------------\n")

        initial_votes = self.collect_votes()
        self.voting_results.append(initial_votes)
        self.current_round = 0

        # Store votes in each agent's memory
        self.record_votes_in_memory(initial_votes)

        # Main simulation rounds
        for i in range(rounds):
            self.current_round = i + 1
            print(f"\n--- Round {self.current_round} ---")
            self.discussion_round(discussion_topic)
            round_votes = self.collect_votes()
            self.voting_results.append(round_votes)
            self.record_votes_in_memory(round_votes)

        # Save the detailed voting results to the specified file path
        self.save_voting_results(results_filepath)

    def record_votes_in_memory(self, round_votes):
        """Record votes in each agent's memory"""
        for agent in self.agents:
            own_vote = round_votes.get(agent.name)
            if own_vote:
                agent.record_vote(self.current_round, own_vote, round_votes)

    def discussion_round(self, topic):
        """Run a discussion round with probabilistic message propagation"""
        # Get communication probabilities from config
        p_same = self.simulation_config.get("communication_probabilities", {}).get("P_same", 0.8)
        p_opposite = self.simulation_config.get("communication_probabilities", {}).get("P_opposite", 0.3)

        # Last round's voting results (for determining message propagation)
        last_round_votes = self.voting_results[-1] if self.voting_results else {}

        # Each agent speaks
        for speaking_agent in self.agents:
            prompt = f"Discuss your views on {topic}. What are the main concerns and benefits you see? Consider any points raised by others that you've heard."
            response = speaking_agent.generate_response(prompt)
            print(f"{speaking_agent.name} says: {response}")
            print("\n----------------------------------------\n")
            # Speaker always records their own statement
            speaking_agent.record_own_statement(prompt, response)

            # Speaker's vote from last round (if available)
            speaker_vote = last_round_votes.get(speaking_agent.name)

            # Determine which other agents hear this message
            for listening_agent in self.agents:
                # Skip self (already recorded above)
                if listening_agent.id == speaking_agent.id:
                    continue

                # Get listener's vote from last round
                listener_vote = last_round_votes.get(listening_agent.name)

                # Determine probability based on voting alignment
                if speaker_vote and listener_vote:  # If both have votes
                    if speaker_vote == listener_vote:
                        # Same voting position
                        probability = p_same
                    else:
                        # Opposite voting position
                        probability = p_opposite
                else:
                    # If we don't have voting info, use average probability
                    probability = (p_same + p_opposite) / 2

                # Probabilistic message propagation
                if random.random() < probability:
                    listening_agent.record_observed_statement(speaking_agent.name, response)
                    print(f"  → {listening_agent.name} heard this message")

            print("\n----------------------------------------\n")

    def collect_votes(self):
        # Retrieve the voting criteria from config
        criteria = self.simulation_config.get("vote_criteria", {})
        round_votes = {}

        vote_prompt = f"Based on the discussion so far about '{self.simulation_config.get('discussion_topic')}', {self.simulation_config.get('voting_question')}"

        print(vote_prompt)
        print("\n----------------------------------------\n")
        for agent in self.agents:
            # Generate a specific voting prompt that incorporates memory

            vote_response = agent.generate_response(vote_prompt)

            # Print the agent's voting response
            print(f"{agent.name} votes: {vote_response}")
            print("----------------------------------------\n")

            # Determine vote based on keywords in response
            vote = None
            for key, keywords in criteria.items():
                for keyword in keywords:
                    if keyword.lower() in vote_response.lower():
                        vote = key
                        break

            # Fallback vote if none of the criteria match
            if vote is None:
                # Look for explicit "support" or "against" in the response
                if "yes" in vote_response.lower():
                    vote = "Support"
                elif "no" in vote_response.lower():
                    vote = "Against"
                else:
                    vote = "Abstain"  # Instead of defaulting to Against, use Abstain

            round_votes[agent.name] = vote
            # Record the voting response in agent's memory
            agent.record_own_statement(vote_prompt, vote_response)

        print("Round Votes:", round_votes)
        return round_votes

    def save_voting_results(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.voting_results, f, indent=2)


if __name__ == "__main__":
    engine = SimulationEngine("config/agents.json", "config/simulation_config.json")
    engine.run_simulation("data/voting_results.json")
