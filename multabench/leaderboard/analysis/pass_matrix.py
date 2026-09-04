"""The dataset x model pass/fail matrix at a fixed delta.

Reads pool_scores_long.csv, writes pass_matrix.csv.
Run standalone: `python -m multabench.leaderboard.analysis.pass_matrix`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, EXTRA_MODELS

_RESULTS = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")
_SCORES_CSV = join(_RESULTS, "pool_scores_long.csv")
_MATRIX_CSV = join(_RESULTS, "pass_matrix.csv")

_STATES = ("no_text", "text_only", "all", "ft")
_FOLDS = range(5)

# The one genuine gap in the source data (a dropped run). Any other missing row must fail.
_KNOWN_MISSING_ROWS = {("TabPFNv2", "REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "ft", 4)}


DELTA_DEFAULT = 0.001


def compute_deltas(scores: pd.DataFrame) -> tuple[float, float]:
    """(Delta_Joint, Delta_Awareness) for one model on one dataset.

        Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only))
        Delta_Awareness = mean(ft)  - mean(all)

    Each state's mean is rounded to 3 decimals before differencing, matching the paper.
    """
    models, datasets = scores["model"].unique(), scores["dataset"].unique()
    assert len(models) == 1 and len(datasets) == 1, (
        f"scores must cover exactly one (model, dataset) pair, "
        f"got models={list(models)} datasets={list(datasets)}"
    )
    model, dataset = models[0], datasets[0]

    expected_rows = {(model, dataset, s, f) for s in _STATES for f in _FOLDS}
    actual_rows = set(zip(scores["model"], scores["dataset"], scores["state"], scores["fold"]))
    missing = expected_rows - actual_rows - _KNOWN_MISSING_ROWS
    extra = actual_rows - expected_rows
    assert not missing, f"Missing row(s) for ({model}, {dataset}): {sorted(missing)}"
    assert not extra, f"Unexpected extra/duplicate row(s) for ({model}, {dataset}): {sorted(extra)}"

    means = scores.groupby("state")["test_score"].mean().round(3)
    delta_joint = means["all"] - max(means["no_text"], means["text_only"])
    delta_awareness = means["ft"] - means["all"]
    return float(delta_joint), float(delta_awareness)


def joint_signal_passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """Joint Signal: the joint frozen model beats both unimodal models. Admits to Full."""
    delta_joint, _ = compute_deltas(scores)
    return delta_joint > delta


def awareness_passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """Task-awareness: fine-tuning the encoder beats freezing it."""
    _, delta_awareness = compute_deltas(scores)
    return delta_awareness > delta


def passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """Core admission: Joint Signal and Task-awareness."""
    delta_joint, delta_awareness = compute_deltas(scores)
    return bool(delta_joint > delta and delta_awareness > delta)


def build_pass_matrix(df: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """dataset x model booleans, NaN where a model has no data for a dataset.

    Left as NaN rather than False so the caller decides how to treat missing data.
    """
    rows = {}
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        rows.setdefault(dataset, {})[model] = passes(sub, delta=delta)
    matrix = pd.DataFrame(rows).T
    matrix.index.name = "dataset"
    return matrix[CURATION_MODELS + EXTRA_MODELS]


if __name__ == "__main__":
    df = pd.read_csv(_SCORES_CSV)
    matrix = build_pass_matrix(df)
    matrix.to_csv(_MATRIX_CSV)
    print(f"Wrote {matrix.shape[0]}x{matrix.shape[1]} pass matrix to {_MATRIX_CSV}")
    print(f"NaN cells (model has no data for that dataset): {matrix.isna().sum().sum()}")
