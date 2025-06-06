import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from analyze import load_sentiment_scores, load_ensemble_sentiment_scores


def read_agent_sentiment_scores(folder, finfo):
    sentiment_scores = load_sentiment_scores(f"{folder}/{finfo}_sentiment_scores.json")
    print(finfo, sentiment_scores)
    # Identify all agents that participated (union of all keys)
    agents = sorted({agent for round_scores in sentiment_scores for agent in round_scores})

    # Create a dictionary to hold each agent's sentiment scores over rounds
    agent_sentiments = {agent: [] for agent in agents}
    for round_data in sentiment_scores:
        for agent in agents:
            # Append the sentiment score for the agent in this round; if missing, use None
            agent_sentiments[agent].append(round_data.get(agent, None))

    rounds = list(range(1, len(sentiment_scores) + 1))

    return agent_sentiments

def plot_single_run(tex_lw=240.71031, ppi=72):

    # create figure and subplots/
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)

    folder = "../data/20250416"

    for i, p in enumerate([0, 0.6, 1.0]):
        agent_sentiments = read_agent_sentiment_scores(folder, f"pineapple_rounds_100_p_{p:.1f}_run_0")
        print(f"Agent Sentiments for p={p}: {agent_sentiments}")

        # Prepare data for violin plot
        agents = list(agent_sentiments.keys())
        sentiment_data = []

        for agent in agents:
            # Filter out None values and ignore first half of the scores
            valid_sentiments = [score for score in agent_sentiments[agent] if score is not None]
            half_point = len(valid_sentiments) // 2
            sentiment_data.append(valid_sentiments[half_point:])

        # Create violin plot
        #plt.violinplot(sentiment_data, positions=np.array(range(len(agents)))-0.25+0.25*i)
        plt.boxplot(sentiment_data, positions=np.array(range(len(agents)))-0.25+0.25*i, widths=0.2, sym='')

    plt.xticks(range(len(agents)), agents, rotation=45, ha='right')
    plt.ylabel('Sentiment Score')
    plt.xlabel('Agents')
    plt.title('Distribution of Sentiment Scores by Agent')
    plt.tight_layout()
    plt.show()

    print(agent_sentiments)


