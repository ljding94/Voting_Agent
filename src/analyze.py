import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd


def load_voting_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_sentiment_scores(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_ensemble_sentiment_scores(folder, finfo, Mrun):
    agent_sentiments_ensemble = []
    agents = None
    for m in range(Mrun):
        filename = f"{folder}/{finfo}_run_{m}_sentiment_scores.json"
        sentiment_scores = load_sentiment_scores(filename)
        # Skip this run if no sentiment data
        if not sentiment_scores:
            print(sentiment_scores)
            print(f"Warning: No sentiment data found in {filename}, skipping run {m}.")
            continue
        if agents is None:
            agents = sorted({agent for round_scores in sentiment_scores for agent in round_scores})
        agent_sentiments = {agent: [] for agent in agents}
        for round_data in sentiment_scores:
            for agent in agents:
                # Append the sentiment score for the agent in this round; if missing, use None
                agent_sentiments[agent].append(round_data.get(agent, None))
        agent_sentiments_ensemble.append(agent_sentiments)
    return agent_sentiments_ensemble


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
    ax1.set_ylim(-6, 6)
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
    ax2.set_ylim(-6, 6)

    plt.tight_layout()
    plt.savefig(f"{folder}/{finfo}_sentiment_scores.png")
    # plt.show()


def plot_ensemble_sentiment_scores(folder, finfo, Mrun):

    # Get ensemble sentiment scores across multiple runs
    agent_sentiments_ensemble = load_ensemble_sentiment_scores(folder, finfo, Mrun)
    print("np.shape(agent_sentiments_ensemble)", np.shape(agent_sentiments_ensemble))
    print("agent_sentiments_ensemble", agent_sentiments_ensemble)

    # Check if we have data
    if not agent_sentiments_ensemble or not agent_sentiments_ensemble[0]:
        print(f"No ensemble sentiment data found for {finfo}")
        return

    # Get the list of agents and number of rounds from the first run
    agents = list(agent_sentiments_ensemble[0].keys())
    rounds = list(range(1, len(agent_sentiments_ensemble[0][agents[0]]) + 1))

    # Calculate mean and std dev for each agent at each round across ensemble runs
    agent_mean_by_round = {agent: [] for agent in agents}
    agent_std_by_round = {agent: [] for agent in agents}

    for round_idx in range(len(rounds)):
        for agent in agents:
            # Collect all values for this agent at this round across all runs
            agent_round_values = [run_data[agent][round_idx] for run_data in agent_sentiments_ensemble if run_data[agent][round_idx] is not None]

            if agent_round_values:
                agent_mean_by_round[agent].append(np.mean(agent_round_values))
                agent_std_by_round[agent].append(np.std(agent_round_values))
            else:
                agent_mean_by_round[agent].append(0)
                agent_std_by_round[agent].append(0)

    # Calculate overall population mean and std dev for each round
    population_mean = []
    population_std = []

    for round_idx in range(len(rounds)):
        round_means = [agent_mean_by_round[agent][round_idx] for agent in agents]
        population_mean.append(np.mean(round_means))
        population_std.append(np.std(round_means))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Plot each agent's mean sentiment with error bands
    for agent in agents:
        ax1.plot(rounds, agent_mean_by_round[agent], marker="o", label=agent, alpha=0.7)
        ax1.fill_between(rounds, np.array(agent_mean_by_round[agent]) - np.array(agent_std_by_round[agent]), np.array(agent_mean_by_round[agent]) + np.array(agent_std_by_round[agent]), alpha=0.2)

    # Also plot the population average with error band
    ax1.plot(rounds, population_mean, color="black", linewidth=2, label="Population Average")
    ax1.fill_between(rounds, np.array(population_mean) - np.array(population_std), np.array(population_mean) + np.array(population_std), alpha=0.3, color="gray", label="Population Std Dev")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Average Sentiment Score")
    ax1.set_title("Ensemble Agent Sentiment Scores Over Rounds")
    ax1.set_xticks(rounds)
    ax1.set_ylim(-6, 6)
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Bar chart of each agent's overall mean sentiment with error bars
    agent_overall_means = [np.mean(agent_mean_by_round[agent]) for agent in agents]
    agent_overall_stds = [np.mean(agent_std_by_round[agent]) for agent in agents]

    ax2.bar(agents, agent_overall_means, yerr=agent_overall_stds, capsize=5, alpha=0.7)
    ax2.set_xlabel("Agent")
    ax2.set_ylabel("Average Sentiment Score")
    ax2.set_title("Ensemble Mean and Std of Each Agent's Sentiment")
    ax2.grid(True, axis="y")
    ax2.set_ylim(-6, 6)

    plt.tight_layout()
    plt.savefig(f"{folder}/ensemble_scores_{finfo}.png")
    # plt.show()


def save_ensemble_sentiment_stats_to_csv(folder, finfo, Mrun):
    # Get ensemble sentiment scores across multiple runs
    agent_sentiments_ensemble = load_ensemble_sentiment_scores(folder, finfo, Mrun)

    # Check if we have data
    if not agent_sentiments_ensemble or not agent_sentiments_ensemble[0]:
        print(f"No ensemble sentiment data found for {finfo}")
        return

    # Get the list of agents from the first run
    agents = list(agent_sentiments_ensemble[0].keys())

    # Create a pandas DataFrame to store the data
    data = []

    # Process each run
    for run_idx, run_data in enumerate(agent_sentiments_ensemble):
        row_data = {}
        all_agent_means = []

        for agent in agents:
            agent_scores = run_data[agent]
            # Calculate stats for the last half of rounds
            half_idx = len(agent_scores) // 2
            last_half_scores = [score for score in agent_scores[half_idx:] if score is not None]

            agent_mean = np.mean(last_half_scores) if last_half_scores else np.nan
            row_data[f"{agent}_mean"] = agent_mean
            row_data[f"{agent}_std"] = np.std(last_half_scores) if last_half_scores else np.nan

            if not np.isnan(agent_mean):
                all_agent_means.append(agent_mean)

        # Calculate overall statistics across all agents
        row_data["overall_mean"] = np.mean(all_agent_means) if all_agent_means else np.nan
        row_data["overall_std"] = np.std(all_agent_means) if all_agent_means else np.nan

        data.append(row_data)

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)

    # Create CSV file path
    csv_path = f"{folder}/ensemble_stats_{finfo}.csv"

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)

    print(f"Saved ensemble sentiment statistics to {csv_path}")


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


def plot_ensemble_stats_vs_p(folder, finfos, ps):
    # read all of the ensemble stats files

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    colormap = plt.get_cmap("rainbow")  # you can change this to another colormap if desired
    colors = colormap(np.linspace(0, 1, 10))  # 10 agents

    # Process each p value and corresponding file info
    for i, (finfo, p) in enumerate(zip(finfos, ps)):
        csv_path = f"{folder}/ensemble_stats_{finfo}.csv"
        df = pd.read_csv(csv_path)

        ax1.plot(
            np.ones(len(df["overall_mean"])) * p,
            df["overall_mean"],
            # yerr=df["overall_std"] / np.sqrt(10),
            marker="o",
            mfc="None",
            ls="None",
            label=f"p={p:.1f}",
            color="black",
            alpha=0.4 + 0.5 * p,
        )

        # Process each agent for the current p value
        for agent_idx, agent in enumerate(sorted([col.split("_")[0] for col in df.columns if "_mean" in col and col != "overall_mean"])):
            # Extract agent metrics
            agent_means = df[f"{agent}_mean"]
            agent_stds = df[f"{agent}_std"]

            # Plot point with error bars
            mk = "o" if agent_idx % 2 == 0 else "s"
            ax2.plot(
                np.ones(len(agent_means)) * p + (agent_idx - 4.5) * 0.01,  # Slight horizontal offset for clarity
                agent_means,
                # yerr=agent_stds / np.sqrt(10),
                marker=mk,
                mfc="None",
                ls="None",
                label=f"{agent}" if i == 0 else "",  # Only label agents in the first iteration
                color=colors[agent_idx % 10],
                alpha=0.6,
            )

    ax1.set_xlabel("p")
    ax1.set_ylabel("Population Sentiment")

    ax2.set_xlabel("p")
    ax2.set_ylabel("Agent Sentiment")
    ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{folder}/ensemble_stats_vs_p.png", dpi=300)
    plt.show()
