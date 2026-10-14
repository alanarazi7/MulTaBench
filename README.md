<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="MulTaBench: Benchmarking Multimodal Tabular Learning" width="640">
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

Default hyperparameters, `--size 10k`, 6 folds, frozen `e5-small` text and `dino-small` image embeddings
(ConTextTab reads raw text itself). Elo is fit on one battle per dataset and model pair, with RandomForest
at 1000 and 95% intervals from bootstrapping the datasets; a model with no result on a dataset has no
battles there. TabPFN-v2 and v2.5 take at most 10 classes, which leaves out 3 of the 80 datasets.

| # | Model | Elo, 80 datasets | 95% CI | Text, 40 | Images, 40 | Datasets |
|---|---|---|---|---|---|---|
| 1 | TabPFN-3.5 | 2176 | +236 / −173 | 2040 | 2405 | 80 |
| 2 | Kumo-Tabular-Small | 2138 | +208 / −148 | 2103 | 2256 | 80 |
| 3 | TabICLv2 | 1756 | +140 / −117 | 1654 | 1941 | 80 |
| 4 | TabPFN-3 | 1674 | +143 / −118 | 1561 | 1872 | 80 |
| 5 | TabPFN-v2p5 | 1665 | +131 / −118 | 1653 | 1745 | 77 |
| 6 | TabPFN-v2 | 1406 | +117 / −97 | 1321 | 1557 | 77 |
| 7 | TabM | 1367 | +98 / −89 | 1348 | 1434 | 80 |
| 8 | ConTextTab | 1352 | +112 / −102 | 1274 | 1492 | 80 |
| 9 | CatBoost | 1297 | +67 / −55 | 1307 | 1319 | 80 |
| 10 | TabDPT | 1253 | +112 / −93 | 1116 | 1457 | 80 |
| 11 | RealMLP | 1082 | +81 / −72 | 1104 | 1059 | 80 |
| 12 | LightGBM | 1057 | +63 / −60 | 1091 | 1013 | 80 |
| 13 | RandomForest | 1000 | anchor | 1000 | 1000 | 80 |
| 14 | XGBoost | 986 | +67 / −64 | 1022 | 936 | 80 |

Each model's runs are in `multabench/leaderboard/models/<model>.csv`, one row per (dataset, fold), failed
runs included. `python -m multabench.leaderboard.elo` prints the table, with win rate, average rank and
normalized error; `--modalities text` or `--modalities image,image-text` restricts it to those datasets.

## Getting started

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env     # Hugging Face token

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --text_encoder e5-small --image_encoder dino-small
```

`pyproject.toml` lists the core dependencies and one extra per model package, so `pip install -e ".[catboost]"`
installs what one model needs; each wrapper imports its package only when it runs. `init.sh` sets up the shared
`.venv` from `requirements.txt`, which locks the core and every extra.

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
may use and its memory limit, and raises before running if any differs; leaderboard runs always pass it. Without it, a
run on other hardware only prints how it differs. On Slurm, request `--gres=gpu:1 --cpus-per-task=8 --mem=32G` on an
H100 node, or `--cpus-per-task=8 --mem=32G` on an 8-core Xeon 8581C node for CPU runs. The sweeps' Slurm jobs run under
`srun`, so a run sees only the CPUs its job asked for, even when Slurm hands it a whole machine.
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
| Leaderboard | Elo via `bencheval`, RandomForest (default) anchored at 1000, missing results imputed with RandomForest and flagged, bootstrapped confidence intervals | Elo as in the paper, RandomForest anchored at 1000, bootstrapped confidence intervals; missing results are left out, not imputed | |
| Splits | Fixed and part of the task definition | Same: a seeded function of the data | ✅ |
| Group and time structure | Audited per dataset; group-aware or forward-in-time splits where needed | Not audited: every dataset is split IID. Some datasets may have group or time structure, in which case IID splits can overestimate performance | |
| Runtime | Train and inference time per 1K rows | Same, with the time spent embedding text and images reported separately | ✅ |
| Hardware | Recorded per run; 8 CPU cores and 32 GB of RAM, with one NVIDIA L40S (48 GB) for GPU models | Recorded per run; every run uses 8 CPU cores and 32 GB of RAM, GPU models with one NVIDIA H100 (80 GB), checked by `benchmark.py --official` | |
| Outer splits | 3 folds, repeated 1–10 times depending on dataset size | 3 folds, repeated twice for every size | |
| Dataset size | Full size | `--size 10k` (the default, and the one the leaderboard reports) caps train and test at 10K and 5K rows; `--size full` keeps every row | |
| Inner validation | 8-fold bagging for every model | No bagging: one fit per split, and models that need validation hold out 10% of the training rows (at most 1,000). This favors models that ensemble internally (TabPFN, TabICL, TabDPT, TabM, RandomForest) over single GBDTs and RealMLP | |
| Hyperparameters | Default, tuned, and tuned + ensembled | Default only | |
| Model implementations | TabArena's model registry | MulTaBench's own wrappers, which add image and text embeddings | |
