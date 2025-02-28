import json
import matplotlib.pyplot as plt

def load_voting_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_voting_results(voting_results):
    # Identify all agents that participated (union of all keys)
    agents = sorted({agent for round_votes in voting_results for agent in round_votes})

    # Create a dictionary to hold each agent's vote over rounds
    agent_votes = {agent: [] for agent in agents}
    for round_data in voting_results:
        for agent in agents:
            # Append the vote for the agent in this round; if missing, use None
            agent_votes[agent].append(round_data.get(agent, None))

    # Map votes to numeric values for plotting: Support -> 1, Against -> 0
    vote_mapping = {"Support": 1, "Against": 0}

    rounds = list(range(1, len(voting_results) + 1))

    plt.figure(figsize=(10, 6))
    for agent, votes in agent_votes.items():
        # Convert votes to numeric values using the mapping
        numeric_votes = [vote_mapping.get(vote, None) for vote in votes]
        plt.plot(rounds, numeric_votes, marker='o', label=agent)

    plt.xlabel("Round")
    plt.ylabel("Vote (Support=1, Against=0)")
    plt.title("Agent Voting Curves Over Rounds")
    plt.xticks(rounds)
    plt.yticks([0, 1], ["Against", "Support"])
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    results_file = "data/custom_voting_results.json"  # Adjust path if needed
    voting_results = load_voting_results(results_file)
    plot_agent_curves(voting_results)
