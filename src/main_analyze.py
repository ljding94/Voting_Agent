from analyze import *


def main():
    folder = "../data/20250416"
    finfo = "pineapple_rounds_100_p_0.0_run_0"
    # plot_sentiment_scores(folder, finfo)
    # return 0
    Mrun = 10

    if 0:
        for p in np.arange(0.0, 1.1, 0.2):
            # for p in [0.0]:
            finfo = f"pineapple_rounds_100_p_{p:.1f}"
            plot_ensemble_sentiment_scores(folder, finfo, Mrun)
            save_ensemble_sentiment_stats_to_csv(folder, finfo, Mrun)
    if 0:
        folder = "../data/20250416"
        finfo = "pineapple_rounds_100_p_0.0_run_0"
        finfos = []
        ps = np.arange(0.0, 1.1, 0.2)
        for p in ps:
            finfos.append(f"pineapple_rounds_100_p_{p:.1f}")
        plot_ensemble_stats_vs_p(folder, finfos, ps)

    if 0:
        folder = "../data/20250428"
        finfos = []
        ps = np.arange(0.1, 1.0, 0.2)
        for p in ps:
            finfo = f"pineapple_rounds_100_p_{p:.1f}"
            plot_ensemble_sentiment_scores(folder, finfo, Mrun)
            save_ensemble_sentiment_stats_to_csv(folder, finfo, Mrun)

            finfos.append(f"pineapple_rounds_100_p_{p:.1f}")
        plot_ensemble_stats_vs_p(folder, finfos, ps)

    if 1:
        folder = "../data/20250416+28"
        finfos = []
        ps = np.arange(0.0, 1.01, 0.1)
        for p in ps:
            finfo = f"pineapple_rounds_100_p_{p:.1f}"

            finfos.append(f"pineapple_rounds_100_p_{p:.1f}")
        plot_ensemble_stats_vs_p(folder, finfos, ps)


if __name__ == "__main__":
    main()
