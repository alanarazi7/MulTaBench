import argparse

from multabench.benchmark.collect import collect_results, write_model_csvs
from multabench.leaderboard.elo import RESULTS_DIR
from multabench.result_keys import STATUS, RunStatus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the run JSONs written by benchmark.py into one CSV per model.")
    parser.add_argument('--output_dir', type=str, default='runs', help="where benchmark.py wrote its JSONs")
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help="default: the leaderboard's results")
    args = parser.parse_args()

    df = collect_results(args.output_dir)
    paths = write_model_csvs(df, args.results_dir)
    n_failed = (df[STATUS] == RunStatus.ERROR).sum()
    print(f"Collected {len(df)} runs ({n_failed} failed) into {len(paths)} files in {args.results_dir}")
