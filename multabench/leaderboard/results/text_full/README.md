# MulTaBench-Full text-tabular runs (uploaded artifacts)

One CSV per dataset, 1,500 runs in total and no failures:

    20 datasets x 5 curation models x 3 states (no_text, text_only, all) x 5 folds

These runs evaluate the **uploaded `multabench-full-*` Kaggle artifacts**, not the original
sources, so they verify both the curation recipes and the upload round-trip. Only the three
Joint-Signal states were run: MulTaBench-Full admission needs no `ft` state, which is what makes
the check cheap (no encoder fine-tuning).

Recompute the admission verdict with:

    python -m multabench.leaderboard.analysis.text_full_verification

## Provenance

The 20 datasets were verified in two batches, which differ in how the CSVs were produced:

| Datasets | Runs | Source |
|---|---|---|
| the first 6 | 450 | wandb sweep `zzlutjez`, exported from the wandb UI |
| the other 14 | 1,050 | wandb sweep `asuqv7o0` (993) + a sharded driver script (57) |

The 14 CSVs are reconstructed from the runs' own logged summaries rather than exported from the
UI, so two columns are weaker than in the first six files and should not be read as exact:

- **`Created`** is derived as the summary timestamp minus the run's `runtime`. Runs log
  `time.localtime()` on hosts set to IDT (+03:00); the value here is converted to UTC.
- **`Sweep`** is empty for 57 of the 1,050 runs. Those runs are genuine, but they were dispatched
  by an explicit driver script rather than a wandb agent: sweep `asuqv7o0` finished with
  TabPFN-2.5 missing on 57 (dataset, state, fold) cells, because that model's checkpoint download
  is licence-gated and one host had no cached copy. The gaps were not a cartesian product, and a
  wandb grid sweep can only express one, so re-running them as a sweep would have meant
  re-running 210 cells and producing 153 duplicate rows.

Every other column is measured, and all 1,050 runs share one git commit (`0e19d58`). The
completeness and duplicate checks in `text_full_verification.py` cover both batches equally.
