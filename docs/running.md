# Running the benchmark

## Adding a model

To put a model on the leaderboard, open a pull request that adds it. Once it is merged, we run it on the leaderboard's
hardware from `master` with `sweep.py --official`, commit its runs to `multabench/leaderboard/models/` and update the
Space. Results run elsewhere are not accepted, so that every row comes from the same hardware and from code anyone can
check out. The pull request adds:

1. A wrapper, `multabench/baselines/<model>.py`: a `TabularModel` subclass with `MODEL_NAME` (the leaderboard's
   name), `SHORT_NAME` (for `--model`), `NEEDS_GPU`, and the `USE_*` flags that pick the shared preprocessing it gets,
   such as the frozen text embeddings, categorical encoding or a validation split. It implements `initialize_model`,
   `fit_model` and, unless the model has scikit-learn's `predict` and `predict_proba`, `predict_from_processed`.
   `multabench/baselines/tabdpt13.py` is a short example.
2. The model in `BASELINES` in `benchmark.py`.
3. Its dependencies as an extra in `pyproject.toml`, with the model's package imported only inside the wrapper's
   methods (`tests/test_registry.py` checks that `benchmark.py` imports without it), and the extra added to
   `requirements.txt` with the `uv pip compile` command in its header. If they conflict with that lock, set `ENV` in
   the wrapper and lock them in `requirements-<ENV>.txt` instead, as LimiX-2 does.

Use the model's default hyperparameters, since the leaderboard doesn't tune, and pin anything downloaded, such as
weights, to a version or commit. Check that `benchmark.py` runs it on a text and an image dataset, and say in the pull
request on which datasets the model can't run, e.g. too many classes, so they go into `EXPECTED_FAILURES` in
`tests/test_leaderboard_results.py`. Add a row to `space/data/models.csv` with the developer, a small square logo in
`space/assets/logos/` and a link to the paper or technical report.

## Environments

`pyproject.toml` lists the core dependencies, a `bench` extra for `benchmark.py` and one extra per model package, so
`pip install -e ".[bench,catboost]"` installs what one model needs; each wrapper imports its package only when it runs.
ConTextTab, Kumo and TabFM are pinned to a git commit, which PyPI doesn't allow in extras, so `requirements-git.txt`
lists them. `init.sh` sets up the shared `.venv` from `requirements.txt`, which locks the core, every extra and those
three. A model whose requirements conflict with that lock declares its own venv in its wrapper (`ENV`), locked in
`requirements-<ENV>.txt` and set up with `source init.sh <ENV>`: Causilo, which needs a newer torch, runs in
`.venv-causilo`, Mitra-v2, which needs AutoGluon 1.6, in `.venv-mitra2`, and LimiX-2 in `.venv-limix2`, on the
Python 3.12 its lock names. `sweep.py` writes each model's venv python into its jobs, so one sweep covers every model.

## One run

`benchmark.py` is the single entry point: it evaluates one model on one dataset fold and writes the
result to `runs/<model>_<dataset>_<text_encoder>_<image_encoder>_<size>_<fold>.json` (`--output_dir` to change), with
`none` for an encoder the dataset has no column for. A run
that fails still writes its JSON, with `"status": "error"` and the error, and exits non-zero; a successful one has
`"status": "ok"`. A run whose JSON already exists is skipped, so a stopped sweep can be relaunched as is; `--overwrite`
reruns it. A model that can't run without a variable (`REQUIRED_ENV`), such as `TABPFN_TOKEN` for TabPFN-v2.5, 3, 3.5
and 3.5-Fast, stops before running and writes no JSON when it is unset, or when the licence server rejects it; jobs on
other machines must be passed it, as they don't read your `.env`.
`--help` lists the available models. `--text_encoder` picks how text columns are embedded: `tfidf`, frozen
`e5-small` (the default) or `e5-large`, or `e5-small-tar` or `e5-large-tar`, E5 fine-tuned on the task with LoRA
(target-aware), or `tile`, [TILE](https://github.com/soda-inria/tile): each row serialized as `The <column> is
<value>. ...`, read by Llama-3.1-8B cut after 14 of its 32 layers, and each text cell's hidden states mean-pooled, so a
cell's embedding sees the rest of its row. TILE embeds live for now and needs the gated Llama licence accepted on the
`HF_TOKEN` account. `--image_encoder` does the same for image columns: frozen `dino-small` (the default) or `dino-large`,
or `dino-small-tar` or `dino-large-tar`. Each applies only to datasets with that kind of column.

## Sweeps

`sweep.py` runs `benchmark.py` over models × datasets × encoders × folds. It writes the runs that
don't have a result yet to a jobs file (`sweep_jobs_<timestamp>.txt`, or `--jobs_file`), one command per line,
which any scheduler can run. `--modalities text`, `image` or `image-text` keeps only those datasets. Text datasets
run once per text encoder and image datasets once per image encoder; image datasets that also have text columns
embed their text with frozen E5 of the image encoder's size: `e5-large` with `dino-large` or `dino-large-tar`, else
`e5-small`. The default encoders include the fine-tuned `e5-small-tar` and `dino-small-tar`, so pass `--text_encoders`/
`--image_encoders` to run fewer. Each run is its own process, and a failed run doesn't stop the others:

```bash
python sweep.py --models light,cat --modalities text --text_encoders e5-small                   # only write the jobs file
python sweep.py --models light,cat --modalities text --text_encoders e5-small --launcher local  # run them here, one after another
python sweep.py --models light,cat --modalities text --text_encoders e5-small --launcher slurm \
    --sbatch="--partition=cpu --cpus-per-task=8 --mem=32G" --submit
python sweep.py --models tabm,real --modalities text --text_encoders e5-small --launcher slurm \
    --sbatch="--partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G" --submit
```

With `--launcher slurm`, each line becomes a task of a Slurm array; `--sbatch="..."` is passed to `sbatch`
as is (the `=` is needed, since the value starts with `--`), and without `--submit` the `sbatch` commands are only
printed. With a launcher, the sweep first downloads the runs' datasets and cached embeddings, since runs that
download the same files at once into a shared Hugging Face cache (e.g. on NFS) can fail. All the runs of a sweep get
the same `--sbatch`, so sweep the runs that need a GPU apart from those that don't (see below), as in the last two
commands; with `--official`, a sweep that mixes them stops. Relaunching a sweep runs only what is still missing, and
with `--rerun_failed` also the runs whose result is an error, e.g. after a Hugging Face rate limit.

## Hardware

Every leaderboard run must use the same hardware, so their times compare: one NVIDIA H100 (80 GB), an Intel Xeon
Platinum 8481C with 8 CPU cores and 32 GB of RAM (`GPU_RUN_HARDWARE` in `multabench/utils/hardware.py`). Runs that need
no GPU, a CPU model (CatBoost, LightGBM, XGBoost, RandomForest) whose text is embedded with TF-IDF or read from the
cache, use an Intel Xeon Platinum 8581C with 8 CPU cores and 32 GB of RAM instead (`CPU_RUN_HARDWARE`). With
`--official` (`sweep.py --official` passes it to every run), `benchmark.py` checks the GPU, the processor, the CPUs it
may use and its memory limit, and raises before running if any differs; leaderboard runs always pass it. On Slurm,
request `--gres=gpu:1 --cpus-per-task=8 --mem=32G` on an H100 node, or `--cpus-per-task=8 --mem=32G` on an 8-core Xeon
8581C node for CPU runs. The sweeps' Slurm jobs run under `srun`, so a run sees only the CPUs its job asked for, even
when Slurm hands it a whole machine.
The cached embeddings are computed on one NVIDIA A100, 6 CPU cores and 32 GB of RAM for now
(`EMBEDDING_HARDWARE`); the hardware they ran on is recorded with them.

## Collecting results

`collect_results.py` gathers the run JSONs into one CSV per model in `multabench/leaderboard/models/`, one row per
run, failed runs included: model, dataset, encoders, size and fold, the status, the metric with `test_score` (higher
is better) and `test_error`, train and inference time per 1K rows apart from embedding, embedding time per 1K rows,
and the hardware. A run already in a model's file is replaced by its rerun; the other rows are kept.

```bash
python collect_results.py --output_dir runs
```

Each model's runs are in `multabench/leaderboard/models/<model>.csv`, one row per (dataset, fold), failed
runs included. `python -m multabench.leaderboard.elo` prints the table, with win rate, average rank, normalized
error and the number of folds each model is best on; `--modalities text` or `--modalities image,image-text` restricts
it to those datasets.

Leaderboard runs pass `--official`, which also requires a clean checkout of a commit on `master`, so the `git`
column of every row names code anyone can check out. The image runs of the first 11 models are the one exception: they
ran at `7f28e55`, on the branch of #148, which was squash-merged unchanged.
