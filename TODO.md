# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Bugs

- [ ] Date columns are read from the Kaggle CSVs as strings, and `fit_date_encoders` only picks up
      `datetime64` columns, so dates fall through to text or categorical features. Detect
      string-encoded dates (date separators plus a parse rate of at least 99%) and convert them
      before the numerical and semantic type detection.
- [ ] Pin `pandas < 3`: pandas 3's default string dtype breaks TabSTAR's dtype detection.

## Embedding cache

- [ ] Create the `multabench` organization on Hugging Face.
- [ ] Add `load_embeddings(dataset, fold, size, encoder)`, shaped like `load_split`, which downloads
      cached embeddings from Hugging Face at a pinned revision.
- [ ] Add a script that computes the embeddings for every (dataset, size, fold, encoder) and uploads
      them. Embeddings are stored at full dimension; PCA stays in the pipeline, fitted per fold.
- [ ] Compute and upload `e5-small`, `e5-small-tar`, `dino-small` and `dino-small-tar` embeddings for
      the benchmark datasets. Frozen encoders don't depend on the split, so they are stored once per
      dataset for every row. Each `-tar` fine-tune sees only its fold's training rows, so it is
      keyed by size and fold.
- [ ] Record the encoding time with each embedding file, and report embedding time per 1K rows on
      the leaderboard separately from the model's train and inference time.
- [ ] Upload the fine-tuned LoRA adapters alongside the embeddings, for reproducibility.
- [ ] Make the model runners read the cached embeddings instead of re-embedding, so one fine-tune per
      fold is shared by every model. Each run records how many features had no cached embedding and
      were encoded live.

## Runs

- [ ] Sweep runner: every model × text encoder in {`tfidf`, `e5-small`, `e5-small-tar`} × image
      encoder in {`dino-small`, `dino-small-tar`} × dataset × fold at `--size 10k`, skipping encoders
      for modalities a dataset doesn't have.
- [ ] A time limit per (model, dataset, fold). Timeouts and errors are written as result rows with a
      `status` and the error, and the sweep can be resumed, skipping jobs that already have a row.
- [ ] Run every leaderboard entry on one fixed GPU type.
- [ ] End-to-end models, which take the raw table without our preprocessing: TabSTAR, ConTextTab and
      AutoGluon multimodal.
- [ ] Collect the run JSONs into one results CSV with `metric`, `test_score`, `test_error`, train and
      inference time per 1K rows, and hardware.
- [ ] Rerun every result CSV under `multabench/leaderboard/results/` with the TabArena metrics; the
      current ones still report AUC and R².

## Leaderboard

- [ ] Elo leaderboard as in TabArena: `bencheval`, RandomForest (default) anchored at 1000, missing
      results imputed with RandomForest and flagged, bootstrapped confidence intervals. Update the
      "Leaderboard" row of the README table.
- [ ] Coverage column (share of (dataset, fold) cells each model completed), plus win rate, average
      rank and improvability as secondary columns.
- [ ] Leaderboard tab reading the new results CSV.
- [ ] README section on submitting a model: open a PR with the model's wrapper (and its
      dependencies); maintainers rerun it on the official hardware, and self-reported results are
      not accepted.
- [ ] Remove the remaining paper analysis code in `multabench/leaderboard/`, keeping only the main
      leaderboard tab.

## Open questions

Helpers copied from tabstar that may not be needed:

- [ ] `densify_objects` (`baselines/preprocessing/sparse.py`): converts pandas sparse columns to
      dense. Does any dataset have sparse columns? If not, remove it.
- [ ] `series_to_dt` (`baselines/preprocessing/dates.py`): strips quotes from date strings,
      re-parses with `errors="coerce"` and drops timezones. Once the datasets store datetime
      columns as timezone-naive `datetime64`, is any of this still needed?
- [ ] `get_device` (`utils/devices.py`): picks the first idle GPU through `nvidia-smi`. Keep it, or
      let the runner set the device explicitly?
- [ ] `CPU_CORES` (`utils/devices.py`): models use at most 8 threads. Keep the cap, or make it a
      setting recorded with each run?

## Later, not needed for the first release

- Mirror the 80 datasets to Hugging Face, and decide whether Kaggle stays as a secondary source.
- Minimal installation for `load_split` and `load_embeddings`.
- Hosting the leaderboard as a Hugging Face Space.
- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- A generic Slurm template, configured only through environment variables, for running the sweep
  on other clusters.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
