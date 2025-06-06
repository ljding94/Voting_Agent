from simulation_engine import SimulationEngine
from analyze import *
from dotenv import load_dotenv
import os
import sys


def main(rounds=10, nIA=1, nID=1, nAA=1, nAD=1, p=0.1, run=0):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    # Specify the path for storing voting results
    folder = "../data/data_pool"

    # Run the simulation
    topic = "pineapple on pizza"  # Example topic for discussion

    finfo = f"pineapple_rounds_{rounds:.0f}_nIA_{nIA:.0f}_nID_{nID:.0f}_nAA_{nAA:.0f}_nAD_{nAD:.0f}_p_{p:.1f}_run_{run}"
    engine = SimulationEngine(api_key, nIA, nID, nAA, nAD, topic, rounds, p)

    engine.run_simulation(folder, finfo)

    # After the simulation, analyze and plot the voting results
    # voting_results = load_voting_results(results_filepath)

    # plot_voting_results(voting_results)
    plot_sentiment_scores(folder, finfo)


if __name__ == "__main__":
    # Check if the user provided command-line arguments for rounds and p
    if len(sys.argv) == 8:
        try:
            rounds = int(sys.argv[1])
            nIA = int(sys.argv[2])
            nID = int(sys.argv[3])
            nAA = int(sys.argv[4])
            nAD = int(sys.argv[5])
            p = float(sys.argv[6])
            run = int(sys.argv[7])

        except ValueError:
            print("Invalid input. Please provide an integer for rounds and a float for p (e.g., python3 main 10 1.0).")
            sys.exit(1)
        print(f"Using provided values: rounds={rounds}, p={p}, run={run}")
        main(rounds, nIA, nID, nAA, nAD, p, run)

        # TODO: allow four types of agent
        # Italian-aggreeble, Italian-disagreeble
        # American-aggreeble, American-disagreeble
        # can specify the number of agent in each type as input
        # study: 1. pure case dependency (4 cases) on p
        # 2. mixed case (2 types, just do 2x2=4 mix) with different p
        # 3. full mix
        # tag agent using IA,ID,AA,AD_0 etc for easy following analysis

    else:
        print("Using default values: rounds=10, p=1.0, run=0")
        # If no arguments are provided, use the default values.
        main()
