# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Embedding cache

- [ ] Create the `multabench` organization on Hugging Face.
- [ ] Add `load_embeddings(dataset, fold, size, encoder)`, shaped like `load_split`, which downloads
      cached embeddings from Hugging Face at a pinned revision.
- [ ] Add a script that computes the embeddings for every (dataset, size, fold, encoder) and uploads
      them. Embeddings are stored at full dimension; PCA stays in the pipeline, fitted per fold.
- [ ] Compute and upload `e5-small` and `e5-small-tar` embeddings for the benchmark datasets. Each
      `e5-small-tar` fine-tune sees only its fold's training rows, so it is keyed by size and fold.
- [ ] Upload the fine-tuned LoRA adapters alongside the embeddings, for reproducibility.
- [ ] Make the model runners read the cached embeddings instead of re-embedding, so one fine-tune per
      fold is shared by every model.

## Runs

- [ ] Sweep runner: every model × text encoder in {`tfidf`, `e5-small`, `e5-small-tar`} × dataset ×
      fold at `--size 10k`.
- [ ] Collect the run JSONs into one results CSV with `metric`, `test_score`, `test_error`, train and
      inference time per 1K rows, and hardware.
- [ ] Rerun every result CSV under `multabench/leaderboard/results/` with the TabArena metrics; the
      current ones still report AUC and R².

## Leaderboard

- [ ] Elo leaderboard as in TabArena: `bencheval`, RandomForest (default) anchored at 1000, missing
      results imputed with RandomForest and flagged, bootstrapped confidence intervals. Update the
      "Leaderboard" row of the README table.
- [ ] Leaderboard tab reading the new results CSV.
- [ ] Remove the remaining paper analysis code in `multabench/leaderboard/`, keeping only the main
      leaderboard tab.

## Open decisions

- [ ] Image encoders for the image datasets on the first leaderboard (`dino-small`, `dino-small-tar`).
- [ ] Whether TabSTAR, ConTextTab and AutoGluon multimodal appear as reference rows.

## Later, not needed for the first release

- Mirror the 80 datasets to Hugging Face, and decide whether Kaggle stays as a secondary source.
- Minimal installation for `load_split` and `load_embeddings`.
- Hosting the leaderboard as a Hugging Face Space.
- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- Group and time structure audit (splits are IID for now; see the README).
