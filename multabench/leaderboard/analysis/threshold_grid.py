"""Sensitivity analysis (b): how robust is the curated set to the consensus ratio `rho` (the
paper's "3/5")? Sweeps rho over the fixed 5-learner curation panel on the 56-dataset text pool.

Run standalone: `python -m multabench.leaderboard.analysis.threshold_grid`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.curation_accept import (
    CURATION_MODELS, DELTA_DEFAULT, accept_set, compute_deltas, load_pool_5model,
)

_OUT_DIR = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")

RHOS = [0.51, 0.6, 0.7, 0.8, 0.9, 1.0]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def rho_sweep_at_k5(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT,
                     rhos=RHOS) -> pd.DataFrame:
    baseline = accept_set(deltas, CURATION_MODELS, delta=delta, rho=0.6)
    baseline_accepted = {d for d, v in baseline.items() if v}

    rows = []
    for rho in rhos:
        result = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
        accepted = {d for d, v in result.items() if v}
        rows.append({
            "rho": rho,
            "min_votes_needed": round(rho * len(CURATION_MODELS), 2),
            "n_accepted": len(accepted),
            "jaccard_vs_rho_0.6": round(_jaccard(accepted, baseline_accepted), 3),
        })
    return pd.DataFrame(rows)


def main():
    df = load_pool_5model()
    deltas = compute_deltas(df)

    rho_curve = rho_sweep_at_k5(deltas)
    rho_curve.to_csv(join(_OUT_DIR, "rho_sweep_at_k5.csv"), index=False)
    print("=== rho sweep, |M|=5 fixed (baseline rho=3/5=0.6) ===")
    print(rho_curve.to_string(index=False))


if __name__ == "__main__":
    main()
