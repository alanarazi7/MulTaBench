<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo.svg" alt="MulTaBench: Benchmarking Multimodal Tabular Learning" width="640">
  </picture>
</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/multabench/leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86%20Leaderboard-live-06214F" alt="Leaderboard"></a>
  <a href="https://arxiv.org/abs/2605.10616"><img src="https://img.shields.io/badge/arXiv-2605.10616-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2605.10616"><img src="https://img.shields.io/badge/NeurIPS%202026-Spotlight-3D6FC4" alt="NeurIPS 2026 Spotlight"></a>
  <a href="https://huggingface.co/multabench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-80-8EB3EA" alt="Datasets"></a>
  <a href="https://pypi.org/project/multabench/"><img src="https://img.shields.io/pypi/v/multabench?color=06214F" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3D6FC4" alt="Python 3.11+">
  <a href="https://github.com/alanarazi7/MulTaBench/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-8EB3EA" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/multabench/leaderboard"><b>🏆 Leaderboard</b></a> ·
  <a href="https://arxiv.org/abs/2605.10616"><b>📄 Paper</b></a> ·
  <a href="https://huggingface.co/multabench"><b>🤗 Datasets</b></a> ·
  <a href="https://huggingface.co/collections/multabench/multabench-6ad47e4de18e12fe3931a551"><b>📚 Collection</b></a> ·
  <a href="https://github.com/alanarazi7/MulTaBench#getting-started"><b>🚀 Getting started</b></a> ·
  <a href="https://github.com/alanarazi7/MulTaBench#adding-a-model"><b>➕ Add a model</b></a>
</p>

A benchmark for multimodal tabular learning: tables whose columns include images or free text, not
just numbers and categories. It is 20 image-tabular and 20 text-tabular curated datasets, with 40
further datasets released alongside them, 80 in total. The benchmark evaluates tabular learners
under a target-aware setting, where the image and text encoders are fine-tuned on the task rather
than used frozen.

| 🗂️ 80 datasets | 🧠 26 models | 🔁 6 folds each | ⚖️ One protocol |
|:---:|:---:|:---:|:---:|
| 40 with text, 40 with images | foundation models, GBDTs, neural nets | 3 folds × 2 repeats, 10K/5K rows | default hyperparameters, same hardware |

> **Work in progress:** MulTaBench is moving towards a living leaderboard, and `master` will change
> gradually on the way there. To reproduce the paper, use the tag
> [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) or the
> [`official-paper-version`](https://github.com/alanarazi7/MulTaBench/tree/official-paper-version) branch.
> The paper's results are kept in `multabench/leaderboard/legacy_results/` while every model is rerun under the
> leaderboard protocol.

## Leaderboard

<p align="center">
  <a href="https://huggingface.co/spaces/multabench/leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86%20Open%20the%20live%20leaderboard-huggingface.co%2Fspaces%2Fmultabench%2Fleaderboard-06214F?style=for-the-badge" alt="Open the live leaderboard"></a>
</p>

The leaderboard lives at [huggingface.co/spaces/multabench/leaderboard](https://huggingface.co/spaces/multabench/leaderboard),
with per-dataset results and cost; `space/` holds its app and data.

Default hyperparameters, `--size 10k`, 6 folds, frozen `e5-small` text and `dino-small` image embeddings
(ConTextTab, TabSTAR and TabSTAR-v1.1 read raw text themselves; TabSTAR loads, for each dataset it was pretrained
on, the checkpoint that held that dataset out, and TabSTAR-v1.1's pretraining left MulTaBench out). TabPFN-v2,
v2.5 and Mitra-v2 take at most 10 classes, which leaves out 3 of the 80 datasets; TabFM and LimiX-2 cover them with
tabpfn-extensions' output coding. Mitra-v2 also runs out of GPU memory on MUL_TEXT_MELBOURNE_AIRBNB, whose 680
features it doesn't reduce since the table has categorical columns.

<details>
<summary><b>How Elo is computed</b></summary>

With TabArena's own evaluation package, [bencheval](https://pypi.org/project/bencheval/),
pinned in the `leaderboard` extra. Every fold is its own battle: on each of a dataset's 6 folds, each pair of models
plays one battle, won by the lower test error on that fold (a tie counts half). The battles are fit as a Bradley–Terry
model, with RandomForest anchored at 1000. The 95% intervals come from 1,000 bootstrap resamples of the datasets, each
drawn with all 6 of its folds, so the intervals reflect how many datasets there are rather than how many folds. They
are bootstrapped before anchoring, so each model's interval is its own, RandomForest's included, rather than relative
to RandomForest. A model with no result on a fold has no battles there; nothing is imputed. Win rate, average rank, normalized
error and best-on count are per fold as well. Until October 2026 the leaderboard, like the paper, played one battle
per dataset on the mean error over its folds. That makes each dataset winner-take-all, which stretches the scale: on
the then 17 models, Kumo-Tabular was at 2670 rather than 2347, with the same ranking.

</details>

| # | Model | Elo, 80 datasets | 95% CI | Text, 40 | Images, 40 | Datasets |
|---|---|---|---|---|---|---|
| 🥇 | **Kumo-Tabular** | 2227 | +88 / −72 | 2243 | 2302 | 80 |
| 🥈 | **LimiX-2** | 2159 | +90 / −78 | 2167 | 2239 | 80 |
| 🥉 | **Kumo-Tabular-Medium** | 2094 | +72 / −60 | 2101 | 2173 | 80 |
| 4 | TabFM | 2017 | +71 / −53 | 2003 | 2113 | 80 |
| 5 | TabPFN-3.5 | 1957 | +64 / −57 | 1882 | 2108 | 80 |
| 6 | Mitra-v2 | 1935 | +72 / −59 | 1869 | 2076 | 76 |
| 7 | Kumo-Tabular-Small | 1910 | +44 / −38 | 1883 | 2014 | 80 |
| 8 | Causilo | 1854 | +43 / −37 | 1800 | 1983 | 80 |
| 9 | TabPFN-3.5-Fast | 1843 | +49 / −46 | 1765 | 1995 | 80 |
| 10 | EXAONE-Tabular | 1776 | +35 / −34 | 1794 | 1832 | 80 |
| 11 | TabICLv2 | 1628 | +33 / −33 | 1552 | 1775 | 80 |
| 12 | Xiaomi-TabLDM | 1623 | +34 / −33 | 1541 | 1776 | 80 |
| 13 | TabPFN-3 | 1564 | +35 / −35 | 1471 | 1728 | 80 |
| 14 | TabPFN-v2p5 | 1552 | +31 / −34 | 1536 | 1629 | 77 |
| 15 | TabDPT-1.3 | 1457 | +56 / −65 | 1373 | 1607 | 80 |
| 16 | TabPFN-v2 | 1360 | +43 / −48 | 1285 | 1488 | 77 |
| 17 | ConTextTab | 1333 | +54 / −62 | 1269 | 1446 | 80 |
| 18 | TabM | 1316 | +39 / −43 | 1298 | 1371 | 80 |
| 19 | TabSTAR-v1.1 | 1275 | +64 / −67 | 1365 | 1179 | 80 |
| 20 | CatBoost | 1261 | +38 / −42 | 1264 | 1280 | 80 |
| 21 | TabDPT | 1219 | +58 / −65 | 1100 | 1376 | 80 |
| 22 | TabSTAR | 1202 | +69 / −73 | 1245 | 1161 | 80 |
| 23 | RealMLP | 1093 | +56 / −64 | 1098 | 1093 | 80 |
| 24 | LightGBM | 1048 | +56 / −72 | 1077 | 1013 | 80 |
| 25 | RandomForest | 1000 | +75 / −86 | 1000 | 1000 | 80 |
| 26 | XGBoost | 992 | +63 / −75 | 1028 | 944 | 80 |


Each model's runs are in `multabench/leaderboard/models/<model>.csv`, one row per (dataset, fold), failed
runs included. `python -m multabench.leaderboard.elo` prints the table, with win rate, average rank, normalized
error and the number of folds each model is best on; `--modalities text` or `--modalities image,image-text` restricts
it to those datasets.

Leaderboard runs pass `--official`, which also requires a clean checkout of a commit on `master`, so the `git`
column of every row names code anyone can check out. The image runs of the first 11 models are the one exception: they
ran at `7f28e55`, on the branch of #148, which was squash-merged unchanged.

## Getting started

### Install

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

### Run the benchmark


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
`.venv-causilo`, Mitra-v2, which needs AutoGluon 1.6, in `.venv-mitra2`, and LimiX-2 in `.venv-limix2`, on the
Python 3.12 its lock names. `sweep.py` writes each model's venv python into its jobs, so one sweep covers every model.

`benchmark.py` is the single entry point: it evaluates one model on one dataset fold and writes the
result to `runs/<model>_<dataset>_<text_encoder>_<image_encoder>_<size>_<fold>.json` (`--output_dir` to change), with
`none` for an encoder the dataset has no column for. A run
that fails still writes its JSON, with `"status": "error"` and the error, and exits non-zero; a successful one has
`"status": "ok"`. A run whose JSON already exists is skipped, so a stopped sweep can be relaunched as is; `--overwrite`
reruns it. `--help` lists the available models. `--text_encoder` picks how text columns are embedded: `tfidf`, frozen
`e5-small` (the default) or `e5-large`, or `e5-small-tar`, E5 fine-tuned on the task with LoRA (target-aware).
`--image_encoder` does the same for image columns: frozen `dino-small` (the default) or `dino-large`, or
`dino-small-tar`. Each applies only to datasets with that kind of column.

### Sweeps

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

### Hardware

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

### Collecting results

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
`tests/test_leaderboard_results.py`.

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

To suggest a dataset for a future release, add it to [`CANDIDATE_DATASETS.md`](CANDIDATE_DATASETS.md) in a pull request.

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

## Citation

```bibtex
@inproceedings{arazi2026multabench,
  title     = {MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image},
  author    = {Arazi, Alan and Shapira, Eilam and Grunblat, Shoham and Ventura, Mor and Hoffer, Elad and
               Blayer, Gioia and Holzm{\"u}ller, David and Purucker, Lennart and Varoquaux, Ga{\"e}l and
               Hutter, Frank and Reichart, Roi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2605.10616},
  archivePrefix = {arXiv}
}
```

## License

The code is under the [MIT license](https://github.com/alanarazi7/MulTaBench/blob/master/LICENSE); each dataset keeps
the terms of its source, linked in `MULTABENCH_SOURCES` in `multabench/datasets/all_datasets.py`.
