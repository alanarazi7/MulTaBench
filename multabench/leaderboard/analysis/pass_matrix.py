"""Builds the canonical dataset x model pass/fail matrix at a fixed delta -- the single
artifact that analyses (1) [model committee] and (3) [quorum size] can both consume, since
both hold delta fixed (analysis (2), the delta sweep, needs the underlying Delta_Joint/
Delta_Awareness values instead, not a boolean already fixed at one delta).

Reads committee_pool.py's pool_scores_long.csv. Depends only on committee_pool.py's
CURATION_MODELS/EXTRA_MODELS (for column ordering).

Run standalone: `python -m multabench.leaderboard.analysis.pass_matrix`
"""
from os.path import dirname, join

import pandas as pd

from multabench.leaderboard.analysis.committee_pool import CURATION_MODELS, EXTRA_MODELS

_RESULTS = join(dirname(__file__), "..", "results", "analysis_curation_sensitivity")
_SCORES_CSV = join(_RESULTS, "pool_scores_long.csv")
_MATRIX_CSV = join(_RESULTS, "pass_matrix.csv")

_STATES = ("no_text", "text_only", "all", "ft")

# The image side names its unimodal conditions `non` and `img`.
IMAGE_JOINT_STATES = ("non", "img", "all")
_FOLDS = range(5)

# The ONE known, genuine gap in the raw source data (a single dropped/failed wandb run),
# confirmed by exhaustively diffing the full expected (model, dataset, state, fold) grid
# against pool_scores_long.csv: exactly this row is missing, nothing else. Hardcoded here so
# it doesn't trip the completeness assertion below; any OTHER missing or duplicate row is
# unexpected and must fail loudly, not be silently averaged over.
_KNOWN_MISSING_ROWS = {("TabPFNv2", "REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO", "ft", 4)}


DELTA_DEFAULT = 0.001


_JOINT_STATES = ("no_text", "text_only", "all")


def state_means(scores: pd.DataFrame, states: tuple = _STATES) -> pd.Series:
    """The per-state mean score of ONE model on ONE dataset, rounded to 3 decimals (the paper's
    reported precision) before any differencing.

    Asserts (loudly, not a warning) that `scores` is exactly complete for that (model, dataset)
    pair over `states` -- no missing rows beyond the one known gap, no duplicates/extras --
    since silently computing a mean over fewer folds would understate variance and bias the
    decision without any visible signal.
    """
    models, datasets = scores["model"].unique(), scores["dataset"].unique()
    assert len(models) == 1 and len(datasets) == 1, (
        f"scores must cover exactly one (model, dataset) pair, "
        f"got models={list(models)} datasets={list(datasets)}"
    )
    model, dataset = models[0], datasets[0]

    expected_rows = {(model, dataset, s, f) for s in states for f in _FOLDS}
    actual_rows = {r for r in zip(scores["model"], scores["dataset"], scores["state"], scores["fold"])
                   if r[2] in states}
    missing = expected_rows - actual_rows - _KNOWN_MISSING_ROWS
    extra = actual_rows - expected_rows
    assert not missing, (
        f"Unexpected missing row(s) for ({model}, {dataset}): {sorted(missing)} -- "
        f"this is a NEW data gap, not the one known/hardcoded case. Investigate before "
        f"trusting pass/fail; do not silently add it to _KNOWN_MISSING_ROWS."
    )
    assert not extra, f"Unexpected extra/duplicate row(s) for ({model}, {dataset}): {sorted(extra)}"

    return scores[scores["state"].isin(states)].groupby("state")["test_score"].mean().round(3)


def compute_deltas(scores: pd.DataFrame) -> tuple[float, float]:
    """(Delta_Joint, Delta_Awareness) for ONE model on ONE dataset, over all 4 states.

        Delta_Joint     = mean(all) - max(mean(no_text), mean(text_only))
        Delta_Awareness = mean(ft)  - mean(all)
    """
    means = state_means(scores)
    delta_joint = means["all"] - max(means["no_text"], means["text_only"])
    delta_awareness = means["ft"] - means["all"]
    return float(delta_joint), float(delta_awareness)


def compute_joint_delta(scores: pd.DataFrame, states: tuple = _JOINT_STATES) -> float:
    """Delta_Joint alone, from the 3 states it needs -- no `ft` runs required.

    Identical arithmetic to compute_deltas()[0], but usable on experiments that never fine-tune
    an encoder (fine-tuning is what makes a curation sweep expensive). Pass IMAGE_JOINT_STATES
    for the image side, whose unimodal conditions are named `non` and `img`.
    """
    structured, unstructured, joint = states
    means = state_means(scores, states=states)
    return float(means[joint] - max(means[structured], means[unstructured]))


def joint_signal_passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """Joint Signal: the joint frozen model beats BOTH unimodal models. This alone is the
    MulTaBench-Full admission condition -- it rejects pure-NLP tasks (the unstructured modality
    suffices) and redundant-text tasks (the structured modality suffices)."""
    delta_joint, _ = compute_deltas(scores)
    return delta_joint > delta


def awareness_passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """Tabular Awareness: fine-tuning the encoder inside the joint model beats freezing it."""
    _, delta_awareness = compute_deltas(scores)
    return delta_awareness > delta


def passes(scores: pd.DataFrame, delta: float = DELTA_DEFAULT) -> bool:
    """MulTaBench-Core admission for one (model, dataset): Joint Signal AND Awareness.

        passes <=> Delta_Joint > delta AND Delta_Awareness > delta
    """
    delta_joint, delta_awareness = compute_deltas(scores)
    return bool(delta_joint > delta and delta_awareness > delta)


def build_pass_matrix(df: pd.DataFrame, delta: float = DELTA_DEFAULT) -> pd.DataFrame:
    """A dataset x model boolean matrix (rows=dataset, columns=model), True iff that model
    passes on that dataset at this delta. NaN where a model has no data at all for a dataset
    (e.g. TabPFNv2/TabPFN-2.5's 2 fully-missing datasets) -- deliberately NOT filled to False
    here, so callers can decide how to treat "no data" (the paper's Accept(D) rule treats it
    as a non-pass, i.e. fillna(False), but that's a caller decision, not baked in).
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
