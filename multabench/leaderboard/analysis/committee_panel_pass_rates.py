"""Per-dataset panel pass-rate analysis (analysis 1, model committee).

For each dataset, draws every possible 5-model panel from that dataset's ELIGIBLE models only
(i.e. models with actual data for it -- a model with no data, such as TabPFNv2/TabPFN-2.5 on
the 2 highly-multiclass datasets they can't be evaluated on, is excluded from the draw pool
rather than counted as having failed). Reports, across all those panels, the % that would
accept the dataset at each quorum threshold (>=3, >=4, =5 of the panel's 5 members), compared
against the original decision (the paper's exact panel: LightGBM+CatBoost+TabM+TabPFNv2+
TabPFN-2.5, 3-of-5 rule).

54 of the 56 pool datasets have all 10 models eligible (C(10,5) = 252 panels each); the 2
TabPFN-ineligible datasets (Wine Review, Spotify Genres -- 30 and 114 target classes) draw
from the 8 remaining eligible models (C(8,5) = 56 panels each).

Run standalone: `python -m multabench.leaderboard.analysis.committee_panel_pass_rates`
"""
from itertools import combinations
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, EXTRA_MODELS
from multabench.leaderboard.main_paper.text_pool import _MULTABENCH_POOL_NAMES

_MATRIX_CSV = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity", "pass_matrix.csv")
_OUT_CSV = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity", "committee_panel_pass_rates.csv")

ALL_MODELS = CURATION_MODELS + EXTRA_MODELS
PANEL_SIZE = 5
RHO = 3 / 5


def load_matrix() -> pd.DataFrame:
    return pd.read_csv(_MATRIX_CSV, index_col="dataset")


def original_decision(matrix: pd.DataFrame) -> pd.Series:
    """The paper's exact panel + 3-of-5 rule, per dataset -- 'accept' or 'reject'."""
    sub = matrix[CURATION_MODELS].fillna(False).astype(bool)
    count = sub.sum(axis=1)
    threshold = RHO * len(CURATION_MODELS)
    return count.apply(lambda c: "accept" if c >= threshold else "reject")


def panel_pass_rates(matrix: pd.DataFrame) -> pd.DataFrame:
    """For every dataset, across all C(n_eligible, 5) panels drawn from ITS OWN eligible
    models, the % of panels that would accept it at each quorum threshold."""
    decision = original_decision(matrix)
    rows = []
    for dataset in matrix.index:
        eligible = [m for m in ALL_MODELS if pd.notna(matrix.loc[dataset, m])]
        panels = list(combinations(eligible, PANEL_SIZE))
        counts = pd.Series([
            matrix.loc[dataset, list(panel)].astype(bool).sum() for panel in panels
        ])
        rows.append({
            "dataset": dataset,
            "original_decision": decision[dataset],
            # True only for the 20 datasets in the final MulTaBench benchmark -- 3 of the 23
            # "accept"-decision datasets were accepted by the pipeline but manually excluded
            # from the final 20 (to match the image-tabular subset's size), so
            # original_decision=="accept" does NOT imply in_multabench.
            "in_multabench": dataset in _MULTABENCH_POOL_NAMES,
            "n_eligible_models": len(eligible),
            "n_panels": len(panels),
            "pct_pass_ge3": round((counts >= 3).mean() * 100, 1),
            "pct_pass_ge4": round((counts >= 4).mean() * 100, 1),
            "pct_pass_eq5": round((counts == 5).mean() * 100, 1),
        })
    return pd.DataFrame(rows).set_index("dataset")


def main():
    matrix = load_matrix()
    table = panel_pass_rates(matrix)
    table.to_csv(_OUT_CSV)
    print(f"Wrote {len(table)} rows to {_OUT_CSV}")

    full = table[table["n_eligible_models"] == 10]
    partial = table[table["n_eligible_models"] < 10]
    print(f"\nFully-eligible datasets (10 models, C(10,5)=252 panels): {len(full)}")
    print(f"Partially-eligible datasets: {len(partial)}")
    print(partial.to_string())

    print("\nAccept-decision datasets with pct_pass_ge3 < 100% (fragile accepts):")
    print(full[(full["original_decision"] == "accept") & (full["pct_pass_ge3"] < 100)]
          [["original_decision", "pct_pass_ge3"]].sort_values("pct_pass_ge3").to_string())

    print("\nReject-decision datasets with pct_pass_ge3 > 0% (borderline rejects):")
    print(full[(full["original_decision"] == "reject") & (full["pct_pass_ge3"] > 0)]
          [["original_decision", "pct_pass_ge3"]].sort_values("pct_pass_ge3", ascending=False).to_string())


if __name__ == "__main__":
    main()
