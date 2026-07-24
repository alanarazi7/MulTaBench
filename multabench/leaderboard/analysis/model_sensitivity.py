"""Sensitivity analysis (a): how robust is the curated dataset set to the choice of the 5
curation learners?

- Leave-one-model-out: drop each of the 5 models, recompute Accept(D) on the remaining 4.
- All combinations: every non-empty subset of the 5 models (2^5 - 1 = 31), same rule.
- Agreement score: pairwise Cohen's kappa + overall Fleiss' kappa on each model's individual
  accept vote, across all 56 pool datasets. Directly answers "could TabPFNv2 and TabPFN-2.5
  vote as a bloc?".
- Extended-model generalization: using the 12-model Frozen/TAR-only pool data, check whether
  Delta_Awareness > delta holds for models beyond the original 5 (task-awareness half only --
  no Delta_Joint data exists for these models on the pool).

Run standalone: `python -m multabench.leaderboard.analysis.model_sensitivity`
"""
from itertools import combinations
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from multabench.leaderboard.analysis.curation_accept import (
    CURATION_MODELS, DELTA_DEFAULT, RHO_DEFAULT,
    accept_set, compute_deltas, compute_deltas_awareness_only,
    load_pool_5model, load_pool_extended_awareness, per_model_pass,
)

_OUT_DIR = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def leave_one_out(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT,
                   rho: float = RHO_DEFAULT) -> pd.DataFrame:
    baseline = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
    baseline_accepted = {d for d, v in baseline.items() if v}

    rows = []
    for dropped in CURATION_MODELS:
        remaining = [m for m in CURATION_MODELS if m != dropped]
        result = accept_set(deltas, remaining, delta=delta, rho=rho)
        accepted = {d for d, v in result.items() if v}
        flipped_in = accepted - baseline_accepted
        flipped_out = baseline_accepted - accepted
        rows.append({
            "dropped_model": dropped,
            "n_accepted": len(accepted),
            "n_flipped_in": len(flipped_in),
            "n_flipped_out": len(flipped_out),
            "flipped_in": ", ".join(sorted(flipped_in)),
            "flipped_out": ", ".join(sorted(flipped_out)),
        })
    return pd.DataFrame(rows)


def all_subsets(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT,
                 rho: float = RHO_DEFAULT) -> pd.DataFrame:
    baseline = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
    baseline_accepted = {d for d, v in baseline.items() if v}

    rows = []
    for k in range(1, len(CURATION_MODELS) + 1):
        for subset in combinations(CURATION_MODELS, k):
            result = accept_set(deltas, subset, delta=delta, rho=rho)
            accepted = {d for d, v in result.items() if v}
            rows.append({
                "subset": "+".join(subset),
                "size": k,
                "n_accepted": len(accepted),
                "jaccard_vs_baseline": round(_jaccard(accepted, baseline_accepted), 3),
                "accepted_datasets": frozenset(accepted),
            })
    return pd.DataFrame(rows)


def dataset_stability(subsets_df: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: fraction of the 31 subsets that accept it."""
    all_datasets = sorted(deltas["dataset"].unique())
    n_subsets = len(subsets_df)
    counts = {d: 0 for d in all_datasets}
    for accepted in subsets_df["accepted_datasets"]:
        for d in accepted:
            counts[d] += 1
    baseline = accept_set(deltas, CURATION_MODELS)
    return pd.DataFrame([
        {"dataset": d, "baseline_accept": baseline[d], "frac_subsets_accept": round(counts[d] / n_subsets, 3)}
        for d in all_datasets
    ]).sort_values(["baseline_accept", "frac_subsets_accept"], ascending=[False, False])


def pairwise_subset_agreement(subsets_df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard agreement between the accepted sets of EVERY pair of the 31 model subsets
    (C(31,2) = 465 pairs) -- generalizes the 5x5 per-model kappa matrix to the full
    combinatorial space of curation panels, rather than anchoring everything to the single
    all-5 baseline (as jaccard_vs_baseline in all_subsets() does). Answers "if you'd picked
    any two different subsets of the 5 models, how much would their accepted sets agree?"
    """
    rows = subsets_df.to_dict("records")
    out = []
    for r1, r2 in combinations(rows, 2):
        out.append({
            "subset_1": r1["subset"], "size_1": r1["size"],
            "subset_2": r2["subset"], "size_2": r2["size"],
            "jaccard": round(_jaccard(r1["accepted_datasets"], r2["accepted_datasets"]), 3),
        })
    return pd.DataFrame(out)


def subset_agreement_by_size(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise Jaccard, grouped by (size_1, size_2) -- e.g. how much do two random
    3-model panels agree on average, vs. a 3-model and a 5-model panel, etc."""
    sym = pd.concat([
        pairwise_df.rename(columns={"size_1": "a", "size_2": "b"}),
        pairwise_df.rename(columns={"size_2": "a", "size_1": "b"}),
    ], ignore_index=True)
    return sym.pivot_table(index="a", columns="b", values="jaccard", aggfunc="mean").round(3)


def stability_distribution(stability_df: pd.DataFrame) -> pd.DataFrame:
    """Bucket each dataset's frac_subsets_accept (from dataset_stability) into how contested
    its decision is across all 31 model subsets."""
    bins = [-0.001, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0]
    labels = ["0% (always rejected)", "1-25%", "26-50%", "51-75%", "76-99%", "100% (always accepted)"]
    bucket = pd.cut(stability_df["frac_subsets_accept"], bins=bins, labels=labels)
    return bucket.value_counts().reindex(labels).rename("n_datasets").reset_index().rename(columns={"index": "stability_bucket"})


def agreement_matrix(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """Pairwise Cohen's kappa between each pair of models' per-dataset accept vote.
    Two datasets (e.g. Spotify Genres, 114 classes) have no TabPFNv2/TabPFN-2.5 runs at all
    (likely too many classes) -- handled pairwise-complete (dropped only for pairs involving
    the missing model), not dropped globally.
    """
    votes = per_model_pass(deltas, delta=delta)
    pivot = votes.pivot(index="dataset", columns="model", values="passes")[CURATION_MODELS]

    matrix = pd.DataFrame(index=CURATION_MODELS, columns=CURATION_MODELS, dtype=float)
    for m1 in CURATION_MODELS:
        for m2 in CURATION_MODELS:
            if m1 == m2:
                matrix.loc[m1, m2] = 1.0
            else:
                pair = pivot[[m1, m2]].dropna().astype(bool)
                matrix.loc[m1, m2] = round(cohen_kappa_score(pair[m1], pair[m2]), 3)
    return matrix


def fleiss_kappa(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT) -> float:
    """Overall Fleiss' kappa across the 5 models' per-dataset accept vote (2 categories).
    Datasets missing any of the 5 models' votes are dropped (Fleiss requires a fixed number
    of raters per item)."""
    votes = per_model_pass(deltas, delta=delta)
    pivot = votes.pivot(index="dataset", columns="model", values="passes")[CURATION_MODELS]
    pivot = pivot.dropna().astype(bool)
    n_items, n_raters = pivot.shape
    counts_true = pivot.sum(axis=1).values
    counts_false = n_raters - counts_true

    p_j_true = counts_true.sum() / (n_items * n_raters)
    p_j_false = counts_false.sum() / (n_items * n_raters)
    P_e = p_j_true ** 2 + p_j_false ** 2

    P_i = (counts_true ** 2 + counts_false ** 2 - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()

    return round((P_bar - P_e) / (1 - P_e), 3) if P_e != 1 else float("nan")


def extended_model_awareness(delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """Fraction of pool datasets with Delta_Awareness > delta, per model, for the 5 curation
    models plus all models available via text_source/more_baselines (Frozen/TAR-only pool)."""
    df = load_pool_extended_awareness()
    deltas = compute_deltas_awareness_only(df)
    rows = []
    for model in sorted(deltas["model"].unique()):
        sub = deltas[deltas["model"] == model]
        n = len(sub)
        n_pass = (sub["delta_awareness"] > delta).sum()
        rows.append({
            "model": model,
            "is_curation_model": model in CURATION_MODELS,
            "n_datasets": n,
            "n_pass_awareness": int(n_pass),
            "frac_pass_awareness": round(n_pass / n, 3) if n else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("frac_pass_awareness", ascending=False)


TABPFN_FAMILY = ["TabPFNv2", "TabPFN-2.5"]
NON_TABPFN_MODELS = ["LightGBM", "CatBoost", "TabM"]
ALT_PANEL_MODELS = ["RandomForest", "RealMLP", "TabDPT", "XGBoost", "TabICLv2"]


def family_swap(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT,
                 rho: float = RHO_DEFAULT) -> pd.DataFrame:
    """Drop the whole TabPFN family (both correlated variants) at once, and the mirror case
    of keeping ONLY the TabPFN family -- a stronger test of the "same-family bloc vote"
    concern than single leave-one-out, since it removes/isolates the correlated pair wholesale
    rather than one model at a time."""
    baseline = accept_set(deltas, CURATION_MODELS, delta=delta, rho=rho)
    baseline_accepted = {d for d, v in baseline.items() if v}

    rows = []
    for label, models in [("Drop TabPFN family (3 remain)", NON_TABPFN_MODELS),
                           ("ONLY TabPFN family (2 models)", TABPFN_FAMILY)]:
        result = accept_set(deltas, models, delta=delta, rho=rho)
        accepted = {d for d, v in result.items() if v}
        rows.append({
            "scenario": label,
            "models": "+".join(models),
            "n_accepted": len(accepted),
            "jaccard_vs_baseline": round(_jaccard(accepted, baseline_accepted), 3),
            "flipped_out": ", ".join(sorted(baseline_accepted - accepted)),
            "flipped_in": ", ".join(sorted(accepted - baseline_accepted)),
        })
    return pd.DataFrame(rows)


def alternative_panel_awareness(delta: float = DELTA_DEFAULT, rho: float = RHO_DEFAULT) -> pd.DataFrame:
    """Compare the original 5 curation models against a fully different 5-model panel (the
    paper's 5 supplementary baselines: RandomForest, RealMLP, TabDPT, XGBoost, TabICLv2) on
    Delta_Awareness alone (Delta_Joint isn't available for non-curation models on the pool,
    so this can't redo the full Accept(D) rule -- it's a partial, Awareness-only proxy for
    "what if we'd picked five completely different learners?")."""
    ext = load_pool_extended_awareness()
    deltas = compute_deltas_awareness_only(ext)

    def _vote(models):
        sub = deltas[deltas["model"].isin(models)].copy()
        sub["passes"] = sub["delta_awareness"] > delta
        counts = sub.groupby("dataset")["passes"].sum()
        n_models = sub.groupby("dataset")["model"].nunique()
        return counts >= rho * n_models

    orig_vote = _vote(CURATION_MODELS)
    alt_vote = _vote(ALT_PANEL_MODELS)
    common = orig_vote.index.intersection(alt_vote.index)
    disagreements = common[orig_vote[common] != alt_vote[common]]

    return pd.DataFrame([
        {"dataset": d, "original_5_panel_passes": bool(orig_vote[d]), "alt_5_panel_passes": bool(alt_vote[d])}
        for d in sorted(disagreements)
    ])


def main():
    df = load_pool_5model()
    deltas = compute_deltas(df)

    loo = leave_one_out(deltas)
    loo.to_csv(join(_OUT_DIR, "model_leave_one_out.csv"), index=False)
    print("=== Leave-one-model-out (baseline: 23 accepted) ===")
    print(loo[["dropped_model", "n_accepted", "n_flipped_in", "n_flipped_out"]].to_string(index=False))

    subsets = all_subsets(deltas)
    subsets_out = subsets.drop(columns=["accepted_datasets"])
    subsets_out.to_csv(join(_OUT_DIR, "model_all_subsets.csv"), index=False)
    print("\n=== All 31 subsets: n_accepted by size ===")
    print(subsets.groupby("size")["n_accepted"].agg(["min", "mean", "max"]).round(1))

    stability = dataset_stability(subsets, deltas)
    stability.to_csv(join(_OUT_DIR, "model_dataset_stability.csv"), index=False)
    n_always_accepted = ((stability["baseline_accept"]) & (stability["frac_subsets_accept"] == 1.0)).sum()
    n_always_rejected = ((~stability["baseline_accept"]) & (stability["frac_subsets_accept"] == 0.0)).sum()
    print(f"\nDatasets accepted under ALL 31 subsets: {n_always_accepted}/23")
    print(f"Datasets rejected under ALL 31 subsets: {n_always_rejected}/33")

    dist = stability_distribution(stability)
    dist.to_csv(join(_OUT_DIR, "model_stability_distribution.csv"), index=False)
    print("\n=== Per-dataset stability distribution (how contested across all 31 subsets) ===")
    print(dist.to_string(index=False))

    pairwise = pairwise_subset_agreement(subsets)
    pairwise.to_csv(join(_OUT_DIR, "model_pairwise_subset_agreement.csv"), index=False)
    by_size = subset_agreement_by_size(pairwise)
    by_size.to_csv(join(_OUT_DIR, "model_subset_agreement_by_size.csv"))
    print(f"\n=== Pairwise agreement across all C(31,2)={len(pairwise)} model-subset pairs ===")
    print(f"Overall mean Jaccard: {pairwise['jaccard'].mean():.3f}  "
          f"median: {pairwise['jaccard'].median():.3f}  "
          f"min: {pairwise['jaccard'].min():.3f}  max: {pairwise['jaccard'].max():.3f}")
    majority_only = pairwise[(pairwise["size_1"] >= 3) & (pairwise["size_2"] >= 3)]
    print(f"Restricted to majority-capable subsets (size>=3, n={len(majority_only)}): "
          f"mean Jaccard = {majority_only['jaccard'].mean():.3f}")
    print("\nMean Jaccard by (subset size, subset size):")
    print(by_size)

    agreement = agreement_matrix(deltas)
    agreement.to_csv(join(_OUT_DIR, "model_agreement_kappa.csv"))
    print("\n=== Pairwise Cohen's kappa (per-model accept vote) ===")
    print(agreement)
    tabpfn_kappa = agreement.loc["TabPFNv2", "TabPFN-2.5"]
    other_pairs = [agreement.loc[m1, m2] for m1, m2 in combinations(CURATION_MODELS, 2)
                   if {m1, m2} != {"TabPFNv2", "TabPFN-2.5"}]
    print(f"\nTabPFNv2 x TabPFN-2.5 kappa: {tabpfn_kappa:.3f}  |  avg of other 9 pairs: {np.mean(other_pairs):.3f}")
    print(f"Fleiss' kappa (all 5 models): {fleiss_kappa(deltas):.3f}")

    extended = extended_model_awareness()
    extended.to_csv(join(_OUT_DIR, "model_extended_awareness.csv"), index=False)
    print("\n=== Extended-model task-awareness generalization (Delta_Awareness > delta only) ===")
    print(extended.to_string(index=False))

    swap = family_swap(deltas)
    swap.drop(columns=["flipped_out", "flipped_in"]).to_csv(join(_OUT_DIR, "model_family_swap.csv"), index=False)
    print("\n=== TabPFN family swap (baseline: 23 accepted with all 5) ===")
    print(swap[["scenario", "models", "n_accepted", "jaccard_vs_baseline"]].to_string(index=False))

    alt_panel = alternative_panel_awareness()
    alt_panel.to_csv(join(_OUT_DIR, "model_alt_panel_disagreements.csv"), index=False)
    print(f"\n=== Original-5 vs. fully-different-5 panel, Delta_Awareness only: "
          f"{len(alt_panel)} disagreements out of 56 ===")
    print(alt_panel.to_string(index=False))


if __name__ == "__main__":
    main()
