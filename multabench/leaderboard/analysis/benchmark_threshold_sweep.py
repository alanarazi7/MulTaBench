"""How much of MulTaBench survives a stricter acceptance bar?

delta_sweep.py and threshold_grid.py ask this of the 56-dataset text candidate pool. This asks
it of the 40 released Core datasets, image and text together, which is the quantity the paper
reports. Same rule, same 5 curation learners, same Delta definitions; only the denominator
differs.

The quorum is taken over the learners that actually ran on a dataset, so the two datasets
TabPFNv2 and TabPFN-2.5 cannot handle are judged by their 3 eligible learners rather than
being disqualified at rho = 1.

Run standalone: `python -m multabench.leaderboard.analysis.benchmark_threshold_sweep`
"""
import glob
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, _MODEL_LABELS
from multabench.leaderboard.analysis.delta_sweep import DELTAS
from multabench.leaderboard.analysis.pass_matrix import DELTA_DEFAULT, compute_deltas, passes_delta
from multabench.leaderboard.analysis.threshold_grid import RHOS

_RESULTS = join(dirname(__file__), "..", "results")
_OUT_DIR = join(_RESULTS, "analysis_curation_sensitivity")
_GRID_CSV = join(_OUT_DIR, "benchmark_threshold_grid.csv")
_DELTAS_CSV = join(_OUT_DIR, "benchmark_deltas.csv")

# The image side names its unimodal conditions `non` and `img`; map them onto the text side's
# names so one Delta computation serves both subsets.
_IMAGE_STATE_ALIASES = {"non": "no_text", "img": "text_only"}

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

    grid = threshold_grid(deltas, rhos=sorted(set(RHOS) | set(RHO_HEADLINE)))
    grid.to_csv(_GRID_CSV, index=False)
    print(f"Wrote {len(grid)} grid rows to {_GRID_CSV}")

    print(f"\n=== Surviving at rho in 3/5, 4/5, 5/5 (baseline delta={DELTA_DEFAULT}) ===")
    headline = grid[grid["rho"].isin([round(r, 3) for r in RHO_HEADLINE])]
    pivot = headline.pivot_table(index=["subset", "delta"], columns="rho", values="n_surviving")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
