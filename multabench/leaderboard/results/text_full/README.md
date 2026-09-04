# MulTaBench-Full text-tabular runs (uploaded artifacts)

`runs.csv` — 1,500 runs, no failures:

    20 datasets x 5 curation models x 3 states (no_text, text_only, all) x 5 folds

Exported from wandb project `tabular_data/multabench-full-text-datasets-playground` and committed
verbatim. **This file is a wandb export and nothing else** — do not hand-edit it, and do not
reconstruct rows from run logs; re-export instead.

These runs evaluate the **uploaded `multabench-full-*` Kaggle artifacts**, not the original
sources, so they verify both the curation recipes and the upload round-trip: a curation bug that
silently dropped or corrupted a text column would show up here as a lost Joint Signal. Only the
three Joint-Signal states were run — MulTaBench-Full admission needs no `ft` state, which is what
makes the check cheap (no encoder fine-tuning).

Recompute the admission verdict with:

    python -m multabench.leaderboard.analysis.text_full_verification

## Reading the file

The export has no `multimodal_state` column, so `text_full_verification.py` recovers the state
from `Name`, which `benchmark.py` builds as `{model}_{dataset}_{state}_{fold}`. Stripping the
known dataset and fold leaves the state exactly; an unexpected state raises rather than being
silently admitted. Adding `multimodal_state` to a future export makes that parse unnecessary.

The runs were produced in three batches, distinguishable only by `Created` (the export carries no
`Sweep` column): sweep `zzlutjez` for the first six datasets, sweep `asuqv7o0` for the other 14,
and a sharded driver script for 57 runs that the second sweep left missing. Those 57 cells were
TabPFN-2.5 runs whose checkpoint download is licence-gated, and one host had no cached copy; the
gaps were not a cartesian product, and a wandb grid sweep can only express one, so re-running them
as a sweep would have meant 210 cells and 153 duplicate rows.
