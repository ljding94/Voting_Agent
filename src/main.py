from simulation_engine import SimulationEngine
from analyze import load_voting_results, plot_voting_results


def main():
    # Specify the path for storing voting results
    results_filepath = "../data/custom_voting_results.json"

    # Run the simulation
    engine = SimulationEngine("config/agents.json", "config/simulation_config.json")
    engine.run_simulation(results_filepath=results_filepath)

    # After the simulation, analyze and plot the voting results
    voting_results = load_voting_results(results_filepath)
    plot_voting_results(voting_results)

if __name__ == "__main__":
    main()
