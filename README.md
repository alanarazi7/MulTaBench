<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo.svg" alt="MulTaBench: Benchmarking Multimodal Tabular Learning" width="640">
  </picture>
</h1>

> **Work in progress:** MulTaBench is moving towards a living leaderboard, and `master` will change
> gradually on the way there. To reproduce the paper, use the tag
> [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) or the
> [`official-paper-version`](https://github.com/alanarazi7/MulTaBench/tree/official-paper-version) branch.
> The paper's results are kept in `multabench/leaderboard/legacy_results/` while every model is rerun under the
> leaderboard protocol.

A benchmark for multimodal tabular learning: tables whose columns include images or free text, not
just numbers and categories. It is 20 image-tabular and 20 text-tabular curated datasets, with 40
further datasets released alongside them, 80 in total. The benchmark evaluates tabular learners
under a target-aware setting, where the image and text encoders are fine-tuned on the task rather
than used frozen.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616) (NeurIPS 2026 Spotlight)  
**Datasets**: [huggingface.co/multabench](https://huggingface.co/multabench)

## Leaderboard

The leaderboard lives at [huggingface.co/spaces/multabench/leaderboard](https://huggingface.co/spaces/multabench/leaderboard),
with per-dataset results and cost; `space/` holds its app and data.

Default hyperparameters, `--size 10k`, 6 folds, frozen `e5-small` text and `dino-small` image embeddings
(ConTextTab, TabSTAR and TabSTAR-v1.1 read raw text themselves; TabSTAR loads, for each dataset it was pretrained
on, the checkpoint that held that dataset out, and TabSTAR-v1.1's pretraining left MulTaBench out). TabPFN-v2 and
v2.5 take at most 10 classes, which leaves out 3 of the 80 datasets.

**How Elo is computed.** With TabArena's own evaluation package, [bencheval](https://pypi.org/project/bencheval/),
pinned in the `leaderboard` extra. Every fold is its own battle: on each of a dataset's 6 folds, each pair of models
plays one battle, won by the lower test error on that fold (a tie counts half). The battles are fit as a Bradley–Terry
model, with RandomForest anchored at 1000. The 95% intervals come from 1,000 bootstrap resamples of the datasets, each
drawn with all 6 of its folds, so the intervals reflect how many datasets there are rather than how many folds. They
are bootstrapped before anchoring, so each model's interval is its own, RandomForest's included, rather than relative
to RandomForest. A model with no result on a fold has no battles there; nothing is imputed. Win rate, average rank, normalized
error and best-on count are per fold as well. Until October 2026 the leaderboard, like the paper, played one battle
per dataset on the mean error over its folds. That makes each dataset winner-take-all, which stretches the scale: on
the then 17 models, Kumo-Tabular was at 2670 rather than 2347, with the same ranking.

| # | Model | Elo, 80 datasets | 95% CI | Text, 40 | Images, 40 | Datasets |
|---|---|---|---|---|---|---|
| 1 | Kumo-Tabular | 2259 | +107 / −86 | 2295 | 2323 | 80 |
| 2 | Kumo-Tabular-Medium | 2115 | +85 / −68 | 2137 | 2187 | 80 |
| 3 | TabFM | 2034 | +79 / −56 | 2025 | 2129 | 80 |
| 4 | TabPFN-3.5 | 1976 | +71 / −64 | 1901 | 2131 | 80 |
| 5 | Kumo-Tabular-Small | 1930 | +51 / −43 | 1906 | 2034 | 80 |
| 6 | Causilo | 1873 | +47 / −40 | 1822 | 2001 | 80 |
| 7 | TabPFN-3.5-Fast | 1856 | +54 / −50 | 1779 | 2010 | 80 |
| 8 | EXAONE-Tabular | 1789 | +38 / −36 | 1814 | 1843 | 80 |
| 9 | TabICLv2 | 1636 | +36 / −34 | 1557 | 1786 | 80 |
| 10 | Xiaomi-TabLDM | 1630 | +37 / −33 | 1545 | 1788 | 80 |
| 11 | TabPFN-3 | 1569 | +37 / −35 | 1474 | 1736 | 80 |
| 12 | TabPFN-v2p5 | 1558 | +33 / −36 | 1542 | 1636 | 77 |
| 13 | TabDPT-1.3 | 1459 | +57 / −65 | 1373 | 1612 | 80 |
| 14 | TabPFN-v2 | 1361 | +43 / −46 | 1286 | 1490 | 77 |
| 15 | ConTextTab | 1333 | +52 / −57 | 1269 | 1447 | 80 |
| 16 | TabM | 1318 | +37 / −40 | 1299 | 1374 | 80 |
| 17 | TabSTAR-v1.1 | 1274 | +61 / −65 | 1363 | 1179 | 80 |
| 18 | CatBoost | 1262 | +36 / −41 | 1265 | 1282 | 80 |
| 19 | TabDPT | 1218 | +56 / −61 | 1099 | 1377 | 80 |
| 20 | TabSTAR | 1200 | +66 / −69 | 1241 | 1160 | 80 |
| 21 | RealMLP | 1093 | +52 / −61 | 1097 | 1093 | 80 |
| 22 | LightGBM | 1048 | +53 / −69 | 1077 | 1013 | 80 |
| 23 | RandomForest | 1000 | +72 / −85 | 1000 | 1000 | 80 |
| 24 | XGBoost | 992 | +62 / −75 | 1028 | 945 | 80 |

Each model's runs are in `multabench/leaderboard/models/<model>.csv`, one row per (dataset, fold), failed
runs included. `python -m multabench.leaderboard.elo` prints the table, with win rate, average rank, normalized
error and the number of folds each model is best on; `--modalities text` or `--modalities image,image-text` restricts
it to those datasets.

Leaderboard runs pass `--official`, which also requires a clean checkout of a commit on `master`, so the `git`
column of every row names code anyone can check out. The image runs of the first 11 models are the one exception: they
ran at `7f28e55`, on the branch of #148, which was squash-merged unchanged.

## Getting started

To use the datasets from another project, install the core, which needs neither torch nor any model package:

```bash
pip install multabench
```

```python
from multabench import Size, list_datasets, load_embeddings, load_published_results, load_split, score
datasets = list_datasets(modality="text", task="regression", tier="core")  # DatasetInfo: id, hf_repo, modality, ...
split = load_split("MUL_IMAGE_PETFINDER", fold=0, size=Size.TEN_K)  # split.x_train, split.y_train, split.x_test, ...
embeddings = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small")
result = score(split.task_type, y_true=y_test, y_pred=proba)  # metric, test_score, test_error, as the leaderboard
published = load_published_results()  # every leaderboard run, in the columns of multabench/leaderboard/models/
```

Each package version reads every dataset and embeddings repo at the commit pinned in
`multabench/datasets/revisions.json`, and ships the leaderboard's runs as of its release. After uploading to one of
those repos, `python -m multabench.datasets.revisions` pins them all to their current `main`.

To run the benchmark:

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env     # Hugging Face token

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --text_encoder e5-small --image_encoder dino-small
```

`pyproject.toml` lists the core dependencies, a `bench` extra for `benchmark.py` and one extra per model package, so
`pip install -e ".[bench,catboost]"` installs what one model needs; each wrapper imports its package only when it runs.
ConTextTab, Kumo and TabFM are pinned to a git commit, which PyPI doesn't allow in extras, so `requirements-git.txt`
lists them. `init.sh` sets up the shared `.venv` from `requirements.txt`, which locks the core, every extra and those
three. A model whose requirements conflict with that lock declares its own venv in its wrapper (`ENV`), locked in
`requirements-<ENV>.txt` and set up with `source init.sh <ENV>`: Causilo, which needs a newer torch, runs in
`.venv-causilo`. `sweep.py` writes each model's venv python into its jobs, so one sweep covers every model.

`benchmark.py` is the single entry point: it evaluates one model on one dataset fold and writes the
result to `runs/<model>_<dataset>_<text_encoder>_<image_encoder>_<size>_<fold>.json` (`--output_dir` to change), with
`none` for an encoder the dataset has no column for. A run
that fails still writes its JSON, with `"status": "error"` and the error, and exits non-zero; a successful one has
`"status": "ok"`. A run whose JSON already exists is skipped, so a stopped sweep can be relaunched as is; `--overwrite`
reruns it. `--help` lists the available models. `--text_encoder` picks how text columns are embedded: `tfidf`, frozen
`e5-small` (the default) or `e5-large`, or `e5-small-tar`, E5 fine-tuned on the task with LoRA (target-aware).
`--image_encoder` does the same for image columns: frozen `dino-small` (the default) or `dino-large`, or
`dino-small-tar`. Each applies only to datasets with that kind of column.

`sweep.py` runs `benchmark.py` over models × datasets × encoders × folds. It writes the runs that
don't have a result yet to a jobs file (`sweep_jobs_<timestamp>.txt`, or `--jobs_file`), one command per line,
which any scheduler can run. `--modalities text`, `image` or `image-text` keeps only those datasets. Text datasets
run once per text encoder and image datasets once per image encoder; image datasets that also have text columns
always embed their text with frozen `e5-small`. The default encoders include the fine-tuned `e5-small-tar` and
`dino-small-tar`, so pass `--text_encoders`/`--image_encoders` to run fewer. Each run is its own process, and a
failed run doesn't stop the others:

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
the same `--sbatch`, so sweep the runs that need a GPU apart from those that don't (see below), as in the last two commands. Relaunching a sweep runs only what is still missing.

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

`collect_results.py` gathers the run JSONs into one CSV per model in `multabench/leaderboard/models/`, one row per
run, failed runs included: model, dataset, encoders, size and fold, the status, the metric with `test_score` (higher
is better) and `test_error`, train and inference time per 1K rows apart from embedding, embedding time per 1K rows,
and the hardware. A run already in a model's file is replaced by its rerun; the other rows are kept.

```bash
python collect_results.py --output_dir runs
```

To use a benchmark split in your own code, `load_split` downloads the dataset and returns the same
train and test rows that `benchmark.py` evaluates on:

```python
from multabench import Size, load_split

split = load_split("MUL_IMAGE_PETFINDER", fold=0, size=Size.TEN_K)
split.x_train, split.y_train, split.x_test, split.y_test, split.task_type, split.image_folder, split.image_column
```

## Cached embeddings

The frozen text and image embeddings are cached on Hugging Face, one public dataset repo per
encoder: `multabench/embeddings-e5-small` and `multabench/embeddings-dino-small`. A frozen encoder
doesn't depend on the split, so each distinct input of a dataset is embedded once, under
`<dataset>/`, in `embeddings.npz`, which maps every input, exactly as the encoder got it (e5's
`"passage: <column>: <value>"` after a run's missing-text fill, or the image file name), to its raw
encoder output (PCA is fitted per run, on the training rows), and `meta.json`, with the dataset's
rows and columns, the encoding time per feature, the hardware it ran on, the hardware it should run
on (`EMBEDDING_HARDWARE` in `multabench/utils/hardware.py`) and how they differ. A run looks its own
inputs up, so an embedding is never paired with another row's text, and an input missing from the
cache, e.g. after the dataset changed, raises. The cache is optional: `benchmark.py` reads it when
the dataset has embeddings for the encoder, and otherwise embeds live, so new encoders and datasets
need no cache to run. Only the frozen `e5-small` text embeddings are read so far. `load_embeddings`
returns them aligned with `load_split`'s rows:

```python
from multabench import Size, load_embeddings

embeddings = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small")
embeddings.train["description"], embeddings.test["description"], embeddings.meta
```

`python -m multabench.embeddings.embed` embeds one dataset, and `python -m
multabench.embeddings.embed_sweep` runs it over encoders × datasets, with the same `--launcher` and
`--sbatch` options as `sweep.py`, and uploads the result with `--upload`:

```bash
python -m multabench.embeddings.embed_sweep --launcher slurm --sbatch="--partition=a100 --gres=gpu:1 --cpus-per-task=6 --mem=32G" --submit
python -m multabench.embeddings.embed_sweep --upload
```

## Datasets

Every dataset is a public dataset repo in the [`multabench`](https://huggingface.co/multabench)
Hugging Face organization, downloaded on demand. Repos are named
`<core|extended>-<img|text>-<reg|cls>-<name>`; `core` and `extended` are two slices of the
benchmark. Each holds `data.parquet` with explicit column types (datetime,
categorical, numeric, string), `metadata.json` with the target, and, for image datasets, the images
in `images-*.zip`.

The paper used the curated CSV copies on the [`chico89`](https://www.kaggle.com/chico89/datasets)
Kaggle account, which the Hugging Face datasets were converted from. The two may differ: the
Hugging Face datasets store dates, categories and numbers as typed columns where the CSVs hold
strings, and a few non-numeric values in numeric columns were mapped (DVM car's `"1 mile"` to 1,
Anime Planet's `"Unknown"` to missing). Scores on the two versions are not directly comparable.

| What you are looking for | Where it is |
|--------------------------|-------------|
| A dataset's Hugging Face repo | the `MulTaBenchDatasetID` values in `multabench/datasets/all_datasets.py` |
| Where a dataset originally comes from | `MULTABENCH_SOURCES` in `multabench/datasets/all_datasets.py` |
| How a dataset was curated | the recipes in `multabench/benchmark/datasets/` at the [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) tag |
| Size, task and feature counts per dataset | `multabench/leaderboard/results/datasets_summary{,_extra}.csv` at the [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) tag |

The 40 datasets released alongside the benchmark were admitted on a weaker criterion and carry no
tier name of their own, so their scores must not be pooled with the 40 MulTaBench datasets.

## Protocol and TabArena

MulTaBench follows the [TabArena](https://tabarena.ai) protocol where doing so is cheap or affects
correctness, and deliberately keeps its own choices where full parity would multiply the compute.
MulTaBench is not an official TabArena leaderboard.

| | TabArena | MulTaBench | Status |
|---|---|---|---|
| Metrics | ROC AUC (binary), log loss (multiclass), RMSE (regression), via AutoGluon's scorers | Same | ✅ |
| Leaderboard | Elo via `bencheval`, one battle per task fold, RandomForest (default) anchored at 1000, missing results imputed with RandomForest and flagged, bootstrapped confidence intervals | Same Elo, one battle per dataset fold; missing results are left out, neither imputed nor dropping the model, so a model is ranked on the datasets it ran on, which its Datasets column shows | |
| Splits | Fixed and part of the task definition | Same: a seeded function of the data | ✅ |
| Group and time structure | Audited per dataset; group-aware or forward-in-time splits where needed | Not audited: every dataset is split IID. Some datasets may have group or time structure, in which case IID splits can overestimate performance | |
| Runtime | Train and inference time per 1K rows | Same, with the time spent embedding text and images reported separately | ✅ |
| Hardware | Recorded per run; 8 CPU cores and 32 GB of RAM, with one NVIDIA L40S (48 GB) for GPU models | Recorded per run; every run uses 8 CPU cores and 32 GB of RAM, GPU models with one NVIDIA H100 (80 GB), checked by `benchmark.py --official` | |
| Outer splits | 3 folds, repeated 1–10 times depending on dataset size | 3 folds, repeated twice for every size | |
| Dataset size | Full size | `--size 10k` (the default, and the one the leaderboard reports) caps train and test at 10K and 5K rows; `--size full` keeps every row | |
| Inner validation | 8-fold bagging for every model | No bagging: one fit per split, and models that need validation hold out 10% of the training rows (at most 1,000). This favors models that ensemble internally (TabPFN, TabICL, TabDPT, TabM, RandomForest) over single GBDTs and RealMLP | |
| Hyperparameters | Default, tuned, and tuned + ensembled | Default only | |
| Model implementations | TabArena's model registry | MulTaBench's own wrappers, which add image and text embeddings | |
