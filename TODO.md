# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Full rerun

Every leaderboard run is rerun from `master` once the fixes below are merged, so all results share one code version.

- [ ] The other fixes that should land before the rerun.
- [ ] Rerun every model with `sweep.py --official`. This also reruns TabDPT with its seeding (#165), and the Kumo
      models, whose timings mix two code versions around the cache fix (#170).

## Embedding cache

- [ ] Decide how a run that reads cached embeddings reports its embedding time. It records almost none
      today; one option is to charge the cached `encode_seconds` from `meta.json`, scaled to the run's
      rows.
- [ ] Re-time the frozen encoders on an H100, as the tuned ones were. Until then, `with_tar_encode_times` in
      `multabench/leaderboard/embedding_cost.py` charges frozen runs the tuned caches' encoding time.

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
- [ ] ConTextTab reads dates natively, but the shared preprocessing expands them first. Add a
      `USE_DATE_TRANSFORMATION` flag that ConTextTab turns off, see the TODO in `abstract_model.py`.
- [ ] RealMLP isn't bit-reproducible on GPU despite `random_state` (0.4–1.6% on a rerun), likely nondeterministic
      CUDA kernels. Check whether `torch.use_deterministic_algorithms` fixes it, or document it.
- [ ] Run the fine-tuned (`-tar`) encoders and AutoGluon multimodal under the leaderboard protocol, which the paper
      also covered.

## Leaderboard

- [ ] Improvability as a secondary column.
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

- Possibly the large encoders, `e5-large` and `dino-large`, frozen and `-tar`, as their own leaderboard entries.
- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
