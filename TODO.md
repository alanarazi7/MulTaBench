# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Light core package

`multabench` should install as a small library, so that another evaluation harness can depend on it
for the data, the splits, the scoring and the published results without installing torch or any
model package. A model then only needs `fit` and `predict`.

- [ ] Shrink `dependencies` to what `load_split`, the metrics and the result files need (`numpy`,
      `pandas`, `pyarrow`, `scikit-learn`, `huggingface_hub`), and move the rest into extras: `embed`
      for the E5 and DINO encoders and their fine-tuning, `baselines` for the shared `TabularModel`
      preprocessing, next to the existing one extra per model. `requirements.txt` keeps locking the
      full `.venv`.
- [ ] Make `import multabench` and `load_split` import only the core: `benchmark/load.py` imports
      `multabench.embeddings.hub`, which pulls in `baselines.preprocessing` and the encoder
      constants. Move `load_embeddings` and its imports apart from `load_split`, and add a test that
      importing `multabench` loads no torch, transformers or model package.
- [ ] Pin a Hugging Face revision per dataset: `download_dataset` calls `snapshot_download` without
      a revision, so a dataset update changes the rows a given `multabench` commit loads.
- [ ] Scoring in the core: `calculate_metric` imports AutoGluon's scorers. Either reimplement ROC
      AUC, log loss and RMSE with scikit-learn, with a test that they match AutoGluon's scores and
      errors (log loss clipping included), or keep `autogluon.core` as a core dependency.
- [ ] A public model protocol: any object with `fit(x, y, task_type, image_folder, image_column)`
      and `predict(x)`, where `predict` returns what `calculate_metric` scores (one probability
      column per class, in label-encoded order, for classification; values for regression). Decide
      where the target is label-encoded, which `TabularModel` does today, so a protocol model gets
      encoded classes and is scored with the same mapping.
- [ ] `evaluate(model, dataset, fold, size)` in the core: `load_split`, `fit`, `predict`,
      `calculate_metric` and the timings, returning one row in the `leaderboard/models/*.csv` schema.
      `benchmark.py` calls it, and the `TabularModel` baselines become implementations of the
      protocol.
- [ ] `load_results()`: the per-model CSVs in `leaderboard/models/` as one DataFrame, shipped as
      package data and importable without streamlit, so a new model can be ranked against the
      published ones on the same splits.
- [ ] `featurize(split, text_encoder, image_encoder)` in the `embed` extra: the frozen-encoder plus
      per-column PCA preprocessing as a standalone step, so a protocol model can use the same features
      as the baselines. It builds on the shared embedding preprocessing below.
- [ ] Release: a version tag, and a README section on installing the core alone
      (`pip install "multabench @ git+https://github.com/alanarazi7/MulTaBench@<tag>"`) with a
      minimal `fit`/`predict` example.

## Embedding cache

- [ ] Compute and upload the `dino-small` embeddings for the benchmark datasets with
      `python -m multabench.embeddings.embed_sweep`.
- [ ] Cache the `e5-small-tar` and `dino-small-tar` embeddings: one fine-tune per (size, fold), each
      seeing only its fold's training rows, uploaded with its LoRA adapter.
- [ ] Make the model runners read the cached `-tar` embeddings too; only the frozen `e5-small` and `dino-small`
      embeddings are read so far.
- [ ] Decide how a run that reads cached embeddings reports its embedding time. It records almost none
      today; one option is to charge the cached `encode_seconds` from `meta.json`, scaled to the run's
      rows.

## Runs

- [ ] Move `benchmark.py`, `sweep.py` and `collect_results.py` into the package, like
      `multabench.embeddings.embed`, so they run with `python -m` without a checkout.
- [ ] Split `benchmark.py` into a few functions in `multabench/benchmark/` (argument parsing,
      encoder resolution, running and writing the result), leaving the script a thin entry point.
- [ ] Run every leaderboard entry on the official hardware.
- [ ] `sweep.py` takes one `--sbatch`, so a sweep mixing CPU and GPU runs has to be split in two by
      hand. Group its runs by `run_hardware` and take a second set of sbatch arguments for CPU runs.
- [ ] TF-IDF runs, deferred: the first release uses one text preprocessing per model, e5-small. Run
      the models that embed text with `--text_encoders tfidf` as their own leaderboard entries;
      until then, sweeps pass `--text_encoders e5-small`, since the default also lists TF-IDF.
- [ ] End-to-end models, which take the raw table without our preprocessing: TabSTAR, ConTextTab and
      AutoGluon multimodal.
- [ ] Rerun what `multabench/leaderboard/legacy_results/` covers with the leaderboard protocol: the
      frozen small encoders are done; the large and fine-tuned (`-tar`) encoders and TabSTAR and
      AutoGluon multimodal are left. Then delete the legacy results with the paper analysis code.

## Per-model dependencies

New models' requirements conflict with the shared `requirements.txt` (torch 2.7.1): Causilo needs torch>=2.13,
Mitra-v2 autogluon.tabular 1.6, LimiX-2 declares python>=3.12 and torch==2.9.1, TabDPT-1.3 tabdpt 1.3 (which can't
load the current TabDPT weights). As in TabArena, `pyproject.toml` has one extra
per model package and each wrapper imports its package lazily; TabArena then keeps a single shared torch, which these
models can't share:

- [ ] One venv per model, `uv pip install -e ".[<model>]"`; `sweep.py` runs each model with its venv's python.
      Packages with wrong declared requirements (LimiX-2's torch pin, EXAONE-Tabular's scikit-learn floor) are
      installed with `--no-deps`, documented in their wrapper.
- [ ] Then add the other TabArena models at their default configs: TabDPT-1.3, LimiX-2, Causilo, TabFM,
      Mitra-v2, EXAONE-Tabular and Xiaomi-TabLDM.
      Mitra-v2's default fine-tunes within 8-fold bagging, and TabArena dropped Xiaomi-TabLDM over a prediction
      bug when a batch has more rows than the training set; both run as they are.

## Leaderboard

- [ ] Decide whether to impute missing results with RandomForest, as TabArena does; the Elo leaves
      them out for now.
- [ ] Improvability as a secondary column.
- [ ] Leaderboard tab reading `multabench/leaderboard/models/`.
- [ ] README section on submitting a model: open a PR with the model's wrapper (and its
      dependencies); maintainers rerun it on the official hardware, and self-reported results are
      not accepted.
- [ ] Remove the remaining paper analysis code in `multabench/leaderboard/`, keeping only the main
      leaderboard tab.

## Shared fine-tuning code

`e5/e5_finetune.py` and `dino/dino_finetune.py` are near copies. Move what they share into
`finetune/`, keeping only the encoder-specific parts (loading, the dataset class, pooling) per
modality:

- [ ] `_get_lora_target_modules` and `_print_trainable_params_per_layer`, which differ only in the
      module names.
- [ ] `E5ForTuning` / `DINOForTuning`: the same backbone + linear head, apart from pooling.
- [ ] `finetune_*_with_lora`: the same seeding, `LoraConfig`, `TrainingArguments`, early stopping
      and `Trainer` loop.
- [ ] `DinoTrainArgs` / `E5TrainArgs`, and the matching `--dino_*` / `--e5_*` flags and kwargs
      dicts in `benchmark.py`.

## Shared embedding preprocessing

`baselines/preprocessing/text_embeddings.py` and `image_embeddings.py` repeat the same per-column
steps: read cached embeddings or encode, then reduce each column to `PCA_COMPONENTS`. Extract a
shared parent, keeping only the encoders themselves (E5, TF-IDF, DINO) per modality:

- [ ] One column-encoder base for `E5ColumnEncoder`, `SkrubColumnEncoder` and `_ScaledPCA`: an
      optional cache, `embed` (cache lookup, else encode) and `transform` (the fitted reduction).
- [ ] `_caches_per_column`, defined twice with only "text" / "image" differing in its message.
- [ ] `PCA_COMPONENTS`, defined in both files, and the fit-PCA-per-column loops in
      `fit_text_encoders_vanilla` and `fit_image_encoders`.
- [ ] Decide on scaling: images get `StandardScaler` before PCA, text doesn't.

## README

- [ ] Rework the README the way the [TabSTAR README](https://github.com/alanarazi7/TabSTAR) is laid out: a row of
      badges (arXiv, NeurIPS 2026, the Hugging Face datasets, the leaderboard Space, the licence once there is
      one), a short welcome and a Resources list, a figure of the benchmark, and a quick start before the details,
      with the reference sections (protocol, cached embeddings, datasets) further down.

## Later, not needed for the first release

- Hosting the leaderboard as a Hugging Face Space.
- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
