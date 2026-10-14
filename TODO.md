# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

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
- [ ] Rerun TabDPT when it next needs rerunning: its committed runs predate its seeding (#165), so a rerun doesn't
      reproduce them exactly.
- [ ] RealMLP isn't bit-reproducible on GPU despite `random_state` (0.4–1.6% on a rerun), likely nondeterministic
      CUDA kernels. Check whether `torch.use_deterministic_algorithms` fixes it, or document it.
- [ ] Kumo-Tabular ran out of the 32 GB RAM on Hearthstone and Wine Review, and Kumo-Tabular-Medium on Spotify
      Genres. Find what grows (likely the cached context) and fix it within the official hardware, else record
      them as failures.
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

- Minimal installation for `load_split` and `load_embeddings`.
- Hosting the leaderboard as a Hugging Face Space.
- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
