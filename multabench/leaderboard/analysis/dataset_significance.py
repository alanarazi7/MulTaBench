"""Is each dataset's TAR gain significant on its own, or only in aggregate?

Per dataset, pairs every learner-fold run of Joint TAR against the same learner-fold run of
Joint Frozen and applies a one-sided paired t-test to the differences. Testing all 40 released
datasets at once is a multiple-comparison problem, so raw p-values are corrected with
Benjamini-Hochberg and significance is read off the corrected values.

This is a different question from the acceptance rule in pass_matrix.py: that rule asks how
many of 5 curation learners each clear a fixed margin, this asks whether the gain is
distinguishable from noise across every learner that ran.

Run standalone: `python -m multabench.leaderboard.analysis.dataset_significance`
"""
from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_RESULTS = Path(dirname(__file__)) / ".." / "results"
_OUT_CSV = join(_RESULTS, "analysis_curation_sensitivity", "dataset_significance.csv")

ALPHA = 0.05


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values: rank the raw values, scale each by n/rank, then enforce
    monotonicity from the largest down so a q-value never falls below a larger one."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


_KEY = ["model", "dataset", "fold", "state"]


def load_paired_scores() -> pd.DataFrame:
    """[model, dataset, subset, fold, state, test_score] for the Frozen and TAR conditions of
    the 40 released datasets, over every learner that ran both.

    Scores are clipped the way the per-dataset results table clips them, so a single badly
    negative R^2 cannot dominate a dataset's mean gain. One more_baselines export repeats
    another's runs verbatim, so identical rows are collapsed; any remaining repeat of a run
    identity is a real ambiguity and fails loudly rather than being averaged away.
    """
    frames = []
    for subset, folder in [("image", "images"), ("text", "text")]:
        core = pd.concat([pd.read_csv(f) for f in sorted((_RESULTS / folder).glob("*.csv"))],
                         ignore_index=True)
        extra = pd.concat([pd.read_csv(f) for f in sorted((_RESULTS / "more_baselines").glob("*.csv"))],
                          ignore_index=True)
        extra = extra[extra["dataset"].isin(set(core["dataset"]))]
        df = pd.concat([core, extra], ignore_index=True)
        df["subset"] = subset
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["model"]).rename(columns={"multimodal_state": "state"})
    out["model"] = out["model"].str.strip()
    out["test_score"] = out["test_score"].clip(lower=-0.1)
    out = out[out["state"].isin(["all", "ft"])][_KEY + ["subset", "test_score"]]
    out = out.drop_duplicates()

    clashing = out.groupby(_KEY).size()
    clashing = clashing[clashing > 1]
    assert clashing.empty, (
        f"{len(clashing)} run identities carry more than one score: {clashing.index.tolist()[:5]}"
    )
    return out.reset_index(drop=True)


def dataset_significance(scores: pd.DataFrame) -> pd.DataFrame:
    """[dataset, subset, n_pairs, n_wins, mean_gain, t, p_raw, p_bh, significant],
    one row per dataset, corrected across all datasets as a single family."""
    rows = []
    for (subset, dataset), sub in scores.groupby(["subset", "dataset"]):
        paired = sub.pivot_table(index=["model", "fold"], columns="state",
                                 values="test_score").dropna(subset=["all", "ft"])
        if paired.empty:
            continue
        gains = paired["ft"] - paired["all"]
        t, p = stats.ttest_1samp(gains, 0, alternative="greater")
        rows.append({
            "dataset": dataset, "subset": subset,
            "n_pairs": len(gains), "n_wins": int((gains > 0).sum()),
            "mean_gain": round(gains.mean(), 4),
            "t": round(float(t), 3), "p_raw": float(p),
        })
    df = pd.DataFrame(rows)
    df["p_bh"] = benjamini_hochberg(df["p_raw"].to_numpy())
    df["significant"] = df["p_bh"] < ALPHA
    return df.sort_values(["subset", "p_bh", "dataset"]).reset_index(drop=True)


def main():
    scores = load_paired_scores()
    df = dataset_significance(scores)
    df.to_csv(_OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {_OUT_CSV}")

    n_sig, n = int(df["significant"].sum()), len(df)
    print(f"\nSignificant after BH at alpha={ALPHA}: {n_sig}/{n}")
    for subset, sub in df.groupby("subset"):
        print(f"  {subset}: {int(sub['significant'].sum())}/{len(sub)}")

    shown = ["dataset", "subset", "n_pairs", "n_wins", "mean_gain", "p_raw", "p_bh"]
    print("\n=== Not significant ===")
    print(df[~df["significant"]][shown].to_string(index=False))


if __name__ == "__main__":
    main()
