"""Sensitivity analysis (c): how robust is the curated set to the acceptance threshold
`delta` (the paper's 0.001 significance margin)?

Fixed M = all 5 curation models, rho = 3/5. Sweeps delta over a range and reports the
accepted-set size, overlap with the delta=0.001 baseline, and which datasets are "borderline"
(their decision flips somewhere in delta in [0, 0.01]).

Run standalone: `python -m multabench.leaderboard.analysis.delta_sweep`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.curation_accept import (
    CURATION_MODELS, RHO_DEFAULT, accept_set, compute_deltas, load_pool_5model,
)

_OUT_DIR = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")

DELTAS = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
# Fine-grained range used only to locate each dataset's flip point.
_BORDERLINE_RANGE = [i / 10000 for i in range(0, 101)]  # 0.0000 .. 0.0100 step 0.0001


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def delta_sweep(deltas: pd.DataFrame, rho: float = RHO_DEFAULT,
                 delta_values=DELTAS) -> pd.DataFrame:
    baseline = accept_set(deltas, CURATION_MODELS, delta=0.001, rho=rho)
    baseline_accepted = {d for d, v in baseline.items() if v}

    rows = []
    for delta in delta_values:
        result = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
        accepted = {d for d, v in result.items() if v}
        rows.append({
            "delta": delta,
            "n_accepted": len(accepted),
            "jaccard_vs_delta_0.001": round(_jaccard(accepted, baseline_accepted), 3),
        })
    return pd.DataFrame(rows)


def borderline_datasets(deltas: pd.DataFrame, rho: float = RHO_DEFAULT) -> pd.DataFrame:
    """For each dataset, find the delta value (within [0, 0.01]) at which its decision flips
    relative to delta=0.001, if any."""
    baseline = accept_set(deltas, CURATION_MODELS, delta=0.001, rho=rho)
    all_datasets = sorted(deltas["dataset"].unique())

    flip_point = {d: None for d in all_datasets}
    for delta in _BORDERLINE_RANGE:
        result = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
        for d in all_datasets:
            if flip_point[d] is None and result[d] != baseline[d]:
                flip_point[d] = delta

    rows = [
        {"dataset": d, "baseline_accept": baseline[d], "flips_at_delta": flip_point[d]}
        for d in all_datasets if flip_point[d] is not None
    ]
    return pd.DataFrame(rows).sort_values("flips_at_delta")


def main():
    df = load_pool_5model()
    deltas = compute_deltas(df)

    sweep = delta_sweep(deltas)
    sweep.to_csv(join(_OUT_DIR, "delta_sweep.csv"), index=False)
    print("=== delta sweep, M=5, rho=3/5 fixed (baseline delta=0.001) ===")
    print(sweep.to_string(index=False))

    borderline = borderline_datasets(deltas)
    borderline.to_csv(join(_OUT_DIR, "delta_sweep_borderline.csv"), index=False)
    print(f"\n=== Borderline datasets (flip somewhere in delta in [0, 0.01]): {len(borderline)}/56 ===")
    print(borderline.to_string(index=False))


if __name__ == "__main__":
    main()
