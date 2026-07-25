"""Analysis (1) of 3 for the rebuttal: sensitivity of the retained dataset set to the choice
of curation MODEL COMMITTEE. Fixed: delta=0.001, rho=3/5 (see analyses 2 and 3 for those).

Built entirely on top of committee_pool.py's pool_scores_long.csv + passes() -- no dependency
on curation_accept.py / model_sensitivity.py.

Answers, from the rebuttal:
    "The paper does not sufficiently establish the robustness of the retained dataset set to
    the ... choice of curation learners." -> leave_one_out(), all_5_of_10_panels()
    "I do not understand why the authors chose two TabPFN variants ... could they effectively
    vote as a bloc under the 3-out-of-5 rule?" -> pairwise_agreement(), drop_or_keep_only()
    "we could also just switch the panel" -> panel_swap()

Run standalone: `python -m multabench.leaderboard.analysis.committee_sensitivity`
"""
from itertools import combinations
from os.path import dirname, join

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, EXTRA_MODELS, passes

_SCORES_CSV = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity", "pool_scores_long.csv")
_OUT_DIR = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")

ALL_MODELS = CURATION_MODELS + EXTRA_MODELS
RHO = 3 / 5
DELTA = 0.001


def load_scores() -> pd.DataFrame:
    return pd.read_csv(_SCORES_CSV)


def per_model_pass(df: pd.DataFrame, delta: float = DELTA) -> pd.DataFrame:
    """[model, dataset, passes] -- one row per (model, dataset) pair present in the pool."""
    rows = [
        {"model": model, "dataset": dataset, "passes": passes(sub, delta=delta)}
        for (model, dataset), sub in df.groupby(["model", "dataset"])
    ]
    return pd.DataFrame(rows)


def accept_set(votes: pd.DataFrame, models, rho: float = RHO) -> set:
    """Accept(D) <=> count(passing models in `models`) >= rho * |models|.
    A model with no row at all for a dataset (e.g. TabPFNv2/TabPFN-2.5's 2 missing datasets)
    counts as not-passing for that dataset, matching the paper's fixed-|M| denominator."""
    models = list(models)
    sub = votes[votes["model"].isin(models)]
    counts = sub.groupby("dataset")["passes"].sum()
    threshold = rho * len(models)
    all_datasets = votes["dataset"].unique()
    return {d for d in all_datasets if counts.get(d, 0) >= threshold}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def leave_one_out(votes: pd.DataFrame, rho: float = RHO) -> pd.DataFrame:
    """Drop each of the 5 curation models in turn, recompute Accept(D) on the remaining 4."""
    baseline = accept_set(votes, CURATION_MODELS, rho)
    rows = []
    for dropped in CURATION_MODELS:
        remaining = [m for m in CURATION_MODELS if m != dropped]
        acc = accept_set(votes, remaining, rho)
        rows.append({
            "dropped_model": dropped,
            "n_accepted": len(acc),
            "jaccard_vs_baseline": round(_jaccard(acc, baseline), 3),
            "flipped_out": ", ".join(sorted(baseline - acc)),
            "flipped_in": ", ".join(sorted(acc - baseline)),
        })
    return pd.DataFrame(rows)


def panel_swap(votes: pd.DataFrame, rho: float = RHO) -> pd.DataFrame:
    """Swap the entire panel: original 5 curation models vs. the 5 other models entirely."""
    baseline = accept_set(votes, CURATION_MODELS, rho)
    rows = []
    for label, models in [("Original 5 (curation panel)", CURATION_MODELS),
                           ("Alternative 5 (other models)", EXTRA_MODELS)]:
        acc = accept_set(votes, models, rho)
        rows.append({
            "scenario": label,
            "models": "+".join(models),
            "n_accepted": len(acc),
            "jaccard_vs_baseline": round(_jaccard(acc, baseline), 3),
            "flipped_out": ", ".join(sorted(baseline - acc)),
            "flipped_in": ", ".join(sorted(acc - baseline)),
        })
    return pd.DataFrame(rows)


def all_5_of_10_panels(votes: pd.DataFrame, rho: float = RHO) -> pd.DataFrame:
    """Accept(D) for EVERY possible 5-model panel drawn from the 10 available models
    (C(10,5) = 252 panels) -- simulates the full combinatorial space."""
    baseline = accept_set(votes, CURATION_MODELS, rho)
    rows = []
    for panel in combinations(ALL_MODELS, 5):
        acc = accept_set(votes, panel, rho)
        rows.append({
            "panel": "+".join(panel),
            "n_curation_models": len(set(panel) & set(CURATION_MODELS)),
            "n_accepted": len(acc),
            "jaccard_vs_baseline": round(_jaccard(acc, baseline), 3),
        })
    return pd.DataFrame(rows)


def pairwise_agreement(votes: pd.DataFrame, models=CURATION_MODELS) -> pd.DataFrame:
    """Cohen's kappa between each pair of models' individual pass/fail vote -- directly tests
    whether TabPFNv2 and TabPFN-2.5 vote as a correlated bloc."""
    pivot = votes[votes["model"].isin(models)].pivot(index="dataset", columns="model", values="passes")
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
            else:
                pair = pivot[[m1, m2]].dropna().astype(bool)
                matrix.loc[m1, m2] = round(cohen_kappa_score(pair[m1], pair[m2]), 3)
    return matrix


def drop_or_keep_only_tabpfn(votes: pd.DataFrame, rho: float = RHO) -> pd.DataFrame:
    """Stronger bloc-vote test than leave-one-out: drop BOTH TabPFN variants at once, and
    the mirror case of keeping ONLY the TabPFN family."""
    baseline = accept_set(votes, CURATION_MODELS, rho)
    tabpfn_family = ["TabPFNv2", "TabPFN-2.5"]
    non_tabpfn = [m for m in CURATION_MODELS if m not in tabpfn_family]
    rows = []
    for label, models in [("Drop TabPFN family (3 remain)", non_tabpfn),
                           ("ONLY TabPFN family (2 models)", tabpfn_family)]:
        acc = accept_set(votes, models, rho)
        rows.append({
            "scenario": label,
            "models": "+".join(models),
            "n_accepted": len(acc),
            "jaccard_vs_baseline": round(_jaccard(acc, baseline), 3),
        })
    return pd.DataFrame(rows)


def main():
    df = load_scores()
    votes = per_model_pass(df)

    baseline = accept_set(votes, CURATION_MODELS)
    print(f"=== Baseline: {len(baseline)} accepted (paper: 23) ===\n")

    loo = leave_one_out(votes)
    loo.to_csv(join(_OUT_DIR, "committee_leave_one_out.csv"), index=False)
    print("=== Leave-one-model-out ===")
    print(loo[["dropped_model", "n_accepted", "jaccard_vs_baseline"]].to_string(index=False))

    agreement = pairwise_agreement(votes)
    agreement.to_csv(join(_OUT_DIR, "committee_pairwise_kappa.csv"))
    print("\n=== Pairwise kappa (5 curation models) ===")
    print(agreement)
    other_pairs = [agreement.loc[m1, m2] for m1, m2 in combinations(CURATION_MODELS, 2)
                   if {m1, m2} != {"TabPFNv2", "TabPFN-2.5"}]
    print(f"TabPFNv2 x TabPFN-2.5: {agreement.loc['TabPFNv2', 'TabPFN-2.5']:.3f}  "
          f"(avg of other 9 pairs: {sum(other_pairs) / len(other_pairs):.3f})")

    bloc = drop_or_keep_only_tabpfn(votes)
    bloc.to_csv(join(_OUT_DIR, "committee_tabpfn_bloc.csv"), index=False)
    print("\n=== TabPFN family drop / keep-only ===")
    print(bloc.to_string(index=False))

    swap = panel_swap(votes)
    swap.drop(columns=["flipped_out", "flipped_in"]).to_csv(join(_OUT_DIR, "committee_panel_swap.csv"), index=False)
    print("\n=== Panel swap: original 5 vs. a fully different 5 ===")
    print(swap[["scenario", "models", "n_accepted", "jaccard_vs_baseline"]].to_string(index=False))

    all_panels = all_5_of_10_panels(votes)
    all_panels.to_csv(join(_OUT_DIR, "committee_all_5of10_panels.csv"), index=False)
    print(f"\n=== All C(10,5)={len(all_panels)} possible panels ===")
    print(f"n_accepted: min={all_panels['n_accepted'].min()} mean={all_panels['n_accepted'].mean():.1f} "
          f"max={all_panels['n_accepted'].max()}")
    print(f"jaccard vs. baseline: min={all_panels['jaccard_vs_baseline'].min():.3f} "
          f"mean={all_panels['jaccard_vs_baseline'].mean():.3f} max={all_panels['jaccard_vs_baseline'].max():.3f}")
    print("\nMean by # curation models in panel:")
    print(all_panels.groupby("n_curation_models")[["n_accepted", "jaccard_vs_baseline"]].mean().round(3))


if __name__ == "__main__":
    main()
