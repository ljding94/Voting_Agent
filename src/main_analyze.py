from analyze import load_voting_results, plot_voting_results

def main():
    # Specify the path to your saved voting results JSON file.
    results_filepath = "../data/custom_voting_results.json"
    voting_results = load_voting_results(results_filepath)
    plot_voting_results(voting_results)

if __name__ == "__main__":
    main()
