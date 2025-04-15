import json
import matplotlib.pyplot as plt
import numpy as np


def load_voting_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_sentiment_scores(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_sentiment_scores(folder, finfo):
    sentiment_scores = load_sentiment_scores(f"{folder}/{finfo}_sentiment_scores.json")

    # Identify all agents that participated (union of all keys)
    agents = sorted({agent for round_scores in sentiment_scores for agent in round_scores})

    # Create a dictionary to hold each agent's sentiment scores over rounds
    agent_sentiments = {agent: [] for agent in agents}
    for round_data in sentiment_scores:
        for agent in agents:
            # Append the sentiment score for the agent in this round; if missing, use None
            agent_sentiments[agent].append(round_data.get(agent, None))

    rounds = list(range(1, len(sentiment_scores) + 1))

    # Calculate overall mean and standard deviation for each round
    sentiment_mean = []
    sentiment_std = []
    for round_data in sentiment_scores:
        round_scores = [score for score in round_data.values() if score is not None]
        if round_scores:
            sentiment_mean.append(np.mean(round_scores))
            sentiment_std.append(np.std(round_scores))
        else:
            sentiment_mean.append(0)  # Default if no scores in a round
            sentiment_std.append(0)

    # Compute per-agent mean and std across rounds
    agent_means = {}
    agent_stds = {}
    for agent, scores in agent_sentiments.items():
        valid_scores = [score for score in scores if score is not None]
        if valid_scores:
            agent_means[agent] = np.mean(valid_scores)
            agent_stds[agent] = np.std(valid_scores)
        else:
            agent_means[agent] = 0
            agent_stds[agent] = 0

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Plot individual agent sentiment scores over rounds
    for agent, sentiments in agent_sentiments.items():
        ax1.plot(rounds, sentiments, marker="o", label=agent, alpha=0.7)

    # Plot overall mean sentiment with standard deviation\n    ax1.plot(rounds, sentiment_mean, color="black", linewidth=2, label="Average Sentiment")
    ax1.plot(rounds, sentiment_mean, color="black", linewidth=2, label="Average Sentiment")
    ax1.fill_between(rounds, np.array(sentiment_mean) - np.array(sentiment_std), np.array(sentiment_mean) + np.array(sentiment_std), alpha=0.3, color="gray", label="Standard Deviation")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Sentiment Score")
    ax1.set_title("Agent Sentiment Scores Over Rounds")
    ax1.set_xticks(rounds)
    ax1.set_ylim(-6,6)
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Bar chart of each agent's mean sentiment with error bars for std
    agent_names = list(agent_means.keys())
    means = [agent_means[agent] for agent in agent_names]
    stds = [agent_stds[agent] for agent in agent_names]
    ax2.bar(agent_names, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_xlabel("Agent")
    ax2.set_ylabel("Sentiment Score")
    ax2.set_title("Mean and Std of Each Agent's Sentiment")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-6,6)

    plt.tight_layout()
    plt.savefig(f"{folder}/{finfo}_sentiment_scores.png")
    #plt.show()


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
        plt.plot(rounds, numeric_votes, marker="o", label=agent)

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
    # voting_results = load_voting_results(results_file)
    # plot_agent_curves(voting_results)
