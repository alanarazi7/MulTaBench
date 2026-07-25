"""Analysis (b) of 3 for the rebuttal: sensitivity of the retained dataset set to the
acceptance-margin threshold delta (paper default: 0.001). Committee membership (a) and
quorum size (c) are covered elsewhere -- see committee_sensitivity.py /
committee_panel_pass_rates.py.

Reuses existing infrastructure rather than recomputing anything from scratch:
build_pass_matrix() (pass_matrix.py) already takes `delta` as a parameter, and
panel_pass_rates() (committee_panel_pass_rates.py) already reports, per dataset, both the
original committee's decision AND the % of all C(n_eligible, 5) panels that would accept it.
This module just calls that same pipeline once per delta value and stacks the results, so a
single long-format CSV gives both views (same exact committee, and aggregated over the ~252
combos) at every delta simultaneously.

Run standalone: `python -m multabench.leaderboard.analysis.committee_delta_sweep`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_panel_pass_rates import panel_pass_rates
from multabench.leaderboard.analysis.pass_matrix import build_pass_matrix

_SCORES_CSV = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity", "pool_scores_long.csv")
_OUT_CSV = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity", "committee_delta_sweep.csv")

DELTA_DEFAULT = 0.001
DELTAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def delta_sweep(df: pd.DataFrame, deltas=DELTAS) -> pd.DataFrame:
    """Long-format: one row per (dataset, delta). Columns from panel_pass_rates() (the
    original-committee decision, in_multabench, accept_8of10, and pct_pass_ge3/4/eq5 across
    all eligible-model panels) plus `delta`."""
    frames = []
    for delta in deltas:
        matrix = build_pass_matrix(df, delta=delta)
        table = panel_pass_rates(matrix).reset_index()
        table.insert(1, "delta", delta)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def main():
    df = pd.read_csv(_SCORES_CSV)
    table = delta_sweep(df)
    table.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {len(table)} rows ({table['dataset'].nunique()} datasets x {table['delta'].nunique()} deltas) to {_OUT_CSV}")

    baseline_row = table[table["delta"] == DELTA_DEFAULT]
    n_baseline = (baseline_row["original_decision"] == "accept").sum()
    print(f"\nBaseline (delta={DELTA_DEFAULT}): {n_baseline} accepted (paper: 23)")

    print("\n=== Accepted count (original 5-model committee) vs. delta ===")
    summary = table.groupby("delta").apply(
        lambda g: (g["original_decision"] == "accept").sum(), include_groups=False
    )
    print(summary.rename("n_accepted").to_string())

    print("\n=== Mean pct_pass_ge3 (across all eligible-model panels) vs. delta ===")
    print(table.groupby("delta")["pct_pass_ge3"].mean().round(1).to_string())


if __name__ == "__main__":
    main()
