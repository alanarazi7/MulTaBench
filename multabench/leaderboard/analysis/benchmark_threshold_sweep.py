"""How much of MulTaBench survives a stricter acceptance bar?

Asks it of the 40 released Core datasets, image and text together. Same rule and same 5
curation learners the paper curated with, re-applied at each (delta, rho).

The scores come from the final benchmark runs, not from the curation-phase runs the
admission decisions were made on, so the baseline row is not a replication of those
decisions: benchmark_baseline_replication.csv records, per dataset, how many learners
vote to accept on these scores.

The quorum is taken over the learners that actually ran on a dataset, so the two datasets
TabPFNv2 and TabPFN-2.5 cannot handle are judged by their 3 eligible learners rather than
being disqualified at rho = 1.

Run standalone: `python -m multabench.leaderboard.analysis.benchmark_threshold_sweep`
"""
import glob
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, _MODEL_LABELS
from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, compute_deltas, passes_delta

_RESULTS = join(dirname(__file__), "..", "results")
_OUT_DIR = join(_RESULTS, "analysis_curation_sensitivity")
_GRID_CSV = join(_OUT_DIR, "benchmark_threshold_grid.csv")
_DELTAS_CSV = join(_OUT_DIR, "benchmark_deltas.csv")
_REPLICATION_CSV = join(_OUT_DIR, "benchmark_baseline_replication.csv")

# The image side names its unimodal conditions `non` and `img`; map them onto the text side's
# names so one Delta computation serves both subsets.
_IMAGE_STATE_ALIASES = {"non": "no_text", "img": "text_only"}

# The published margin is also the smallest the criterion can express: fold means are rounded
# to 3 decimals, so anything below it accepts a joint model that merely ties its unimodal best.
DELTAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
RHOS = [0.51, 0.6, 0.7, 0.8, 0.9, 1.0]
RHO_HEADLINE = [3 / 5, 4 / 5, 5 / 5]


def load_benchmark_scores() -> pd.DataFrame:
    """[model, dataset, state, fold, test_score, subset] for the 40 Core datasets, 5 curation
    learners, all 4 conditions."""
    frames = []
    for subset, folder in [("image", "images"), ("text", "text")]:
        files = glob.glob(join(_RESULTS, folder, "*.csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df = df.dropna(subset=["model"]).copy()
        df["model"] = df["model"].str.strip().map(_MODEL_LABELS)
        df = df[df["model"].isin(CURATION_MODELS)]
        df = df.rename(columns={"multimodal_state": "state"})
        df["state"] = df["state"].replace(_IMAGE_STATE_ALIASES)
        df["subset"] = subset
        frames.append(df[["model", "dataset", "state", "fold", "test_score", "subset"]])
    out = pd.concat(frames, ignore_index=True)
    out["fold"] = out["fold"].astype(int)
    out["test_score"] = out["test_score"].astype(float)
    return out.sort_values(["subset", "dataset", "model", "state", "fold"]).reset_index(drop=True)


def benchmark_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    """[dataset, subset, model, delta_joint, delta_awareness], one row per (dataset, learner)."""
    rows = []
    for (subset, dataset, model), sub in scores.groupby(["subset", "dataset", "model"]):
        delta_joint, delta_awareness = compute_deltas(sub)
        rows.append({"dataset": dataset, "subset": subset, "model": model,
                     "delta_joint": delta_joint, "delta_awareness": delta_awareness})
    return pd.DataFrame(rows).sort_values(["subset", "dataset", "model"]).reset_index(drop=True)


def surviving(deltas: pd.DataFrame, delta: float, rho: float) -> set:
    """The datasets still accepted at (delta, rho), quorum taken over eligible learners."""
    votes = [passes_delta(j, delta) and passes_delta(a, delta)
             for j, a in zip(deltas["delta_joint"], deltas["delta_awareness"])]
    tally = deltas.assign(passes=votes).groupby("dataset")["passes"].agg(["sum", "count"])
    return {d for d, (n_pass, n_eligible) in tally.iterrows() if n_pass >= rho * n_eligible}


def baseline_replication(deltas: pd.DataFrame, delta: float = DELTA_DEFAULT,
                         rho: float = 3 / 5) -> pd.DataFrame:
    """[dataset, subset, n_pass, n_eligible, margin, accepted]: whether each released dataset
    still clears the quorum on the benchmark runs, and by how much the closest non-voting
    learner missed."""
    rows = []
    for (subset, dataset), sub in deltas.groupby(["subset", "dataset"]):
        votes = [passes_delta(j, delta) and passes_delta(a, delta)
                 for j, a in zip(sub["delta_joint"], sub["delta_awareness"])]
        shortfalls = [min(j, a) for j, a, v in zip(sub["delta_joint"], sub["delta_awareness"], votes)
                      if not v]
        rows.append({
            "dataset": dataset, "subset": subset,
            "n_pass": sum(votes), "n_eligible": len(votes),
            "closest_miss": round(max(shortfalls), 3) if shortfalls else None,
            "accepted": sum(votes) >= rho * len(votes),
        })
    return pd.DataFrame(rows).sort_values(["subset", "n_pass", "dataset"]).reset_index(drop=True)


def threshold_grid(deltas: pd.DataFrame, delta_values=DELTAS, rhos=RHOS) -> pd.DataFrame:
    rows = []
    for subset in ["image", "text", "all"]:
        sub = deltas if subset == "all" else deltas[deltas["subset"] == subset]
        n_total = sub["dataset"].nunique()
        for delta in delta_values:
            for rho in rhos:
                rows.append({"subset": subset, "delta": delta, "rho": round(rho, 3),
                             "n_surviving": len(surviving(sub, delta, rho)), "n_total": n_total})
    return pd.DataFrame(rows)


def main():
    scores = load_benchmark_scores()
    deltas = benchmark_deltas(scores)
    deltas.to_csv(_DELTAS_CSV, index=False)
    print(f"Wrote {len(deltas)} (dataset, learner) deltas over "
          f"{deltas['dataset'].nunique()} datasets to {_DELTAS_CSV}")

    replication = baseline_replication(deltas)
    replication.to_csv(_REPLICATION_CSV, index=False)
    n_ok = int(replication["accepted"].sum())
    print(f"Wrote {len(replication)} replication rows to {_REPLICATION_CSV}")
    print(f"Baseline (delta={DELTA_DEFAULT}, rho=3/5) reproduces {n_ok}/{len(replication)} admissions; "
          f"misses: {replication.loc[~replication['accepted'], 'dataset'].tolist()}")

    grid = threshold_grid(deltas, rhos=sorted(set(RHOS) | set(RHO_HEADLINE)))
    grid.to_csv(_GRID_CSV, index=False)
    print(f"Wrote {len(grid)} grid rows to {_GRID_CSV}")

    print(f"\n=== Surviving at rho in 3/5, 4/5, 5/5 (baseline delta={DELTA_DEFAULT}) ===")
    headline = grid[grid["rho"].isin([round(r, 3) for r in RHO_HEADLINE])]
    pivot = headline.pivot_table(index=["subset", "delta"], columns="rho", values="n_surviving")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
