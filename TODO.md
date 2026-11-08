# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Embedding cache

- [ ] Re-time the frozen encoders on an H100, as the tuned ones were. Until then, `with_tar_encode_times` in
      `multabench/leaderboard/embedding_cost.py` charges frozen runs the tuned caches' encoding time.

## Runs

- [ ] TILE: run the 30 missing runs. On Melbourne Airbnb, TabICLv2's and TabLDM's column embeddings of 8 ensemble
      members at once take 47 GB, past the RAM even offloaded to disk (#281); fewer members at once would keep them on
      the GPU. Profile LimiX-2 there, and Kumo's GPU memory in fit on Spotify Genres.
- [ ] TF-IDF: rerun Mitra-v2 on OSHA Injury, folds 0-4, which the HF Hub's rate limit failed; its Elo skips
      those 5 folds until then.
- [ ] CASE: add CASE-Mitra-v2, still running, and profile Kumo-Tabular's GPU memory in fit on Spotify Genres.
      Ablations later: its authors suggest more than 32 PCA components, and its 50-token cell cut and 16,000-token
      context may hurt the long-text datasets.
- [ ] TILE: revisit its settings per model; every wrapper takes TILE's defaults (30 PCA components per string column,
      no native categories), where the PFNs may do better with the feature window and categories its authors select.

- [ ] Move `benchmark.py`, `sweep.py` and `collect_results.py` into the package, like
      `multabench.embeddings.embed`, so they run with `python -m` without a checkout.
- [ ] Split `benchmark.py` into a few functions in `multabench/benchmark/` (argument parsing,
      encoder resolution, running and writing the result), leaving the script a thin entry point.
- [ ] `sweep.py` takes one `--sbatch`, so a sweep mixing CPU and GPU runs has to be split in two by
      hand. Group its runs by `run_hardware` and take a second set of sbatch arguments for CPU runs.
- [ ] AutoGluon multimodal, the last end-to-end model, which takes the raw table without our preprocessing; the paper
      covered it.
- [ ] ConTextTab reads dates natively, but the shared preprocessing expands them first. Add a
      `USE_DATE_TRANSFORMATION` flag that ConTextTab turns off, see the TODO in `abstract_model.py`.

## Leaderboard

- [ ] Improvability as a secondary column.
- [ ] Improve the Pareto plot and the cost reporting, as the Elo plot now joins each encoder's dots across the models.
- [ ] Once NVIDIA publishes a technical report for Kumo-Tabular, link and cite it in place of its Hugging Face blog
      post: its three models' link in `space/data/models.csv` and its BibTeX on the Cite & Data tab.
- [ ] Update the Space from `master` with a GitHub Action: rerun `multabench.leaderboard.export` and upload
      `space/` when `multabench/leaderboard/models/` changes. It is uploaded by hand for now.

## Core package

- [ ] A public model protocol, any object with `fit` and `predict`, and `evaluate(model, dataset, fold, size)` in the
      core: `load_split`, `fit`, `predict`, `score` and the timings, returning one row in the
      `leaderboard/models/*.csv` schema. `benchmark.py` calls it, and the `TabularModel` baselines implement it.
- [ ] `featurize(split, text_encoder, image_encoder)` in the `bench` extra: the frozen-encoder and per-column PCA
      preprocessing as a standalone step, so a protocol model gets the baselines' features. It builds on the shared
      embedding preprocessing below.

## README

- [ ] A figure of the benchmark near the top, the way the [TabSTAR README](https://github.com/alanarazi7/TabSTAR) has
      one; the badges, quick links and citation are in (#207).

## Later, not needed for the first release

- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
