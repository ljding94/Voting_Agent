from simulation_engine import SimulationEngine
from analyze import *
from dotenv import load_dotenv
import os
import sys

def main(rounds=10, p=1.0, run=0):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    # Specify the path for storing voting results
    folder = "../data/data_pool"
    # Run the simulation
    topic = "pineapple on pizza"  # Example topic for discussion
    finfo = f"pieneapple_rounds_{rounds:.0f}_p_{p:.1f}_run_{run}"
    engine = SimulationEngine(api_key, "config/agents.json",topic, rounds, p)

    engine.run_simulation(folder, finfo)

    # After the simulation, analyze and plot the voting results
    # voting_results = load_voting_results(results_filepath)

    # plot_voting_results(voting_results)
    plot_sentiment_scores(folder, finfo)


if __name__ == "__main__":
    # Check if the user provided command-line arguments for rounds and p
    if len(sys.argv) == 4:
        try:
            rounds = int(sys.argv[1])
            p = float(sys.argv[2])
            run = int(sys.argv[3])
        except ValueError:
            print("Invalid input. Please provide an integer for rounds and a float for p (e.g., python3 main 10 1.0).")
            sys.exit(1)
        print(f"Using provided values: rounds={rounds}, p={p}, run={run}")
        main(rounds, p, run)
    else:
        print("Using default values: rounds=10, p=1.0, run=0")
        # If no arguments are provided, use the default values.
        main()
