# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Embedding cache

- [ ] Compute and upload the `e5-small-tar` and `dino-small-tar` embeddings from `master`:
      `python -m multabench.embeddings.embed_sweep --encoders e5-small-tar,dino-small-tar`, one job per dataset and
      fold, then `--upload`. Then run the leaderboard models with them.
- [ ] Decide how a run that reads cached embeddings reports its embedding time. It records almost none
      today; one option is to charge the cached `encode_seconds` (and a tuned encoder's `finetune_seconds`) from
      `meta.json`, scaled to the run's rows.
- [ ] Decide on scaling before PCA: image embeddings get a `StandardScaler`, text embeddings don't
      (`ColumnEncoder.SCALE`).

## Runs

- [ ] Move `benchmark.py`, `sweep.py` and `collect_results.py` into the package, like
      `multabench.embeddings.embed`, so they run with `python -m` without a checkout.
- [ ] Split `benchmark.py` into a few functions in `multabench/benchmark/` (argument parsing,
      encoder resolution, running and writing the result), leaving the script a thin entry point.
- [ ] `sweep.py` takes one `--sbatch`, so a sweep mixing CPU and GPU runs has to be split in two by
      hand. Group its runs by `run_hardware` and take a second set of sbatch arguments for CPU runs.
- [ ] TF-IDF runs, deferred: the first release uses one text preprocessing per model, e5-small. Run
      the models that embed text with `--text_encoders tfidf` as their own leaderboard entries;
      until then, sweeps pass `--text_encoders e5-small`, since the default also lists TF-IDF.
- [ ] AutoGluon multimodal, the last end-to-end model, which takes the raw table without our preprocessing.
- [ ] Rerun TabDPT when it next needs rerunning: its committed runs predate its seeding (#165), so a rerun doesn't
      reproduce them exactly.
- [ ] RealMLP isn't bit-reproducible on GPU despite `random_state` (0.4–1.6% on a rerun), likely nondeterministic
      CUDA kernels. Check whether `torch.use_deterministic_algorithms` fixes it, or document it.
- [ ] Rerun the Kumo models when they next need rerunning: only their 24 out-of-memory runs postdate the cache fix
      (#170), which also speeds up fit, so their timings mix two code versions.
- [ ] Rerun what `multabench/leaderboard/legacy_results/` covers with the leaderboard protocol: the
      frozen small encoders and TabSTAR are done; the large and fine-tuned (`-tar`) encoders and
      AutoGluon multimodal are left. Then delete the legacy results with the paper analysis code.

## Leaderboard

- [ ] Improvability as a secondary column.
- [ ] Update the Space from `master` with a GitHub Action: rerun `multabench.leaderboard.export` and upload
      `space/` when `multabench/leaderboard/models/` changes. It is uploaded by hand for now.
- [ ] Move the leaderboard out of the README once the Space is announced.
- [ ] README section on submitting a model: open a PR with the model's wrapper (and its
      dependencies); maintainers rerun it on the official hardware, and self-reported results are
      not accepted.
- [ ] Remove the remaining paper analysis code in `multabench/leaderboard/`, keeping only the main
      leaderboard tab.

## Core package

- [ ] A public model protocol, any object with `fit` and `predict`, and `evaluate(model, dataset, fold, size)` in the
      core: `load_split`, `fit`, `predict`, `score` and the timings, returning one row in the
      `leaderboard/models/*.csv` schema. `benchmark.py` calls it, and the `TabularModel` baselines implement it.
- [ ] `featurize(split, text_encoder, image_encoder)` in the `bench` extra: the frozen-encoder and per-column PCA
      preprocessing as a standalone step, so a protocol model gets the baselines' features, built on
      `ColumnEncoder`.

## README

- [ ] Rework the README the way the [TabSTAR README](https://github.com/alanarazi7/TabSTAR) is laid out: a row of
      badges (arXiv, NeurIPS 2026, the Hugging Face datasets, the leaderboard Space, the licence once there is
      one), a short welcome and a Resources list, a figure of the benchmark, and a quick start before the details,
      with the reference sections (protocol, cached embeddings, datasets) further down.

## Later, not needed for the first release

- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
