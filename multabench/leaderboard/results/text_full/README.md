# MulTaBench-Full text-tabular runs (uploaded artifacts)

One CSV per dataset, exported from wandb sweep `zzlutjez`
(`tabular_data/multabench-full-text-datasets-playground`), 450 runs in total and no failures:

    6 datasets x 5 curation models x 3 states (no_text, text_only, all) x 5 folds

These runs evaluate the **uploaded `multabench-full-*` Kaggle artifacts**, not the original
sources, so they verify both the curation recipes and the upload round-trip. Only the three
Joint-Signal states were run: MulTaBench-Full admission needs no `ft` state, which is what makes
the check cheap (no encoder fine-tuning).

Recompute the admission verdict with:

    python -m multabench.leaderboard.analysis.text_full_verification
