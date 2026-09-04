"""Pre-registered random draw of the 4 regression datasets to convert to classification.

MulTaBench-Full's text half targets 10 classification / 10 regression among its 20 extras. Six
extras are already classification; four more come from reformulating a regression target as
equal-frequency bins (see target_bins.py for the per-dataset bin count, itself sampled once and
frozen).

More than four datasets are expected to keep Joint Signal after binning, so a rule is needed to
pick among them. Ranking by margin would select the datasets whose reformulation happens to look
best on this particular sweep -- a selection effect on top of a curation decision. Instead the
four are drawn uniformly at random from whichever datasets pass.

**The seed and the procedure are fixed here BEFORE the sweep's results are known.** That is the
whole point: a seed chosen after seeing the outcome is not a random draw, it is a ranking with
extra steps. Do not adjust SELECTION_SEED to change which datasets come out.
"""
from random import Random
from typing import List

SELECTION_SEED = 20260902
N_CONVERTERS = 4


def draw_converters(admitted: List[str], n: int = N_CONVERTERS) -> List[str]:
    """Draw `n` datasets uniformly at random from those that passed Joint Signal when binned.

    `admitted` is sorted first so the draw depends only on the SET of passing datasets, not on
    the order they happened to finish in the sweep.
    """
    pool = sorted(set(admitted))
    assert len(pool) >= n, (
        f"Only {len(pool)} dataset(s) passed Joint Signal when binned, need {n}: {pool}. "
        f"Binning more datasets, or admitting fewer converters, is a decision to make "
        f"explicitly -- do not lower n to make this pass."
    )
    return Random(SELECTION_SEED).sample(pool, n)


# --- Stratified draw (the rule actually used) ----------------------------------------------
#
# ADOPTED AFTER THE SWEEP'S RESULTS WERE KNOWN, at the user's request. Recorded plainly because
# changing a selection rule post hoc is exactly the kind of thing that has to be visible:
#
#   - The pre-registered rule above was a uniform draw over all passing datasets. All 14 kept
#     Joint Signal when binned (13 finalized at the time of the draw, 9 of them unanimously),
#     so the pool was every candidate rather than a handful of survivors.
#   - A uniform draw of 4 from 14 can land several datasets in the same bin-count bucket by
#     chance -- worst case four C=2 median splits, which is the weakest version of this
#     reformulation to publish and is what the C sampling was meant to avoid.
#   - Stratifying fixes the bin-count spread (one dataset per C) while leaving WHICH dataset
#     comes from each bucket random. It constrains the shape of the result, not its content:
#     no dataset was chosen for scoring better than another.
#
# Only finalized datasets are eligible -- a dataset still mid-sweep has no verdict, so it cannot
# be admitted. Everything not drawn stays a regression task.
def draw_converters_stratified(admitted_bins: dict) -> dict:
    """Draw one dataset per bin-count bucket from those that passed Joint Signal when binned.

    `admitted_bins` maps dataset name -> its frozen bin count C. Datasets are sorted within each
    bucket so the draw depends only on the SET of passing datasets per bucket, and buckets are
    drawn in ascending C order, so the result is reproducible from SELECTION_SEED alone.

    Returns {C: dataset name}.
    """
    buckets = {}
    for dataset, c in admitted_bins.items():
        buckets.setdefault(c, []).append(dataset)
    rng = Random(SELECTION_SEED)
    return {c: rng.choice(sorted(buckets[c])) for c in sorted(buckets)}
