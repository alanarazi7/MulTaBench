import argparse

from multabench.benchmark.collect import collect_results
from multabench.result_keys import STATUS, RunStatus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect the run JSONs written by benchmark.py into one CSV.")
    parser.add_argument('--output_dir', type=str, default='runs', help="where benchmark.py wrote its JSONs")
    parser.add_argument('--csv', type=str, default='results.csv')
    args = parser.parse_args()

    df = collect_results(args.output_dir)
    df.to_csv(args.csv, index=False)
    n_failed = (df[STATUS] == RunStatus.ERROR).sum()
    print(f"Wrote {len(df)} runs ({n_failed} failed) to {args.csv}")
