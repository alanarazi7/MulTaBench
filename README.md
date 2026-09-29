# MulTaBench

> **Work in progress:** MulTaBench is moving towards a living leaderboard, and `master` will change
> gradually on the way there. To reproduce the paper, use the tag
> [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) or the
> [`official-paper-version`](https://github.com/alanarazi7/MulTaBench/tree/official-paper-version) branch.

A benchmark for multimodal tabular learning: tables whose columns include images or free text, not
just numbers and categories. It is 20 image-tabular and 20 text-tabular curated datasets, with 40
further datasets released alongside them, 80 in total. The benchmark evaluates tabular learners
under a target-aware setting, where the image and text encoders are fine-tuned on the task rather
than used frozen.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616) (NeurIPS 2026 Spotlight)  
**Datasets**: [kaggle.com/chico89](https://www.kaggle.com/chico89/datasets)

## Getting started

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env     # Hugging Face and Kaggle credentials

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --multimodal_state all
```

`benchmark.py` is the single entry point: it evaluates one model on one dataset fold and writes the
result to `runs/<model>_<dataset>_<state>_<fold>.json` (`--output_dir` to change). `--help` lists
the available models; `--multimodal_state` is `all` (frozen encoders) or `ft` (the dataset's image
or text encoder fine-tuned on the task).

## Datasets

Every dataset is hosted on the [`chico89`](https://www.kaggle.com/chico89/datasets) Kaggle account
and downloaded on demand; the benchmark datasets are slugged `multabench-<name>` and the ones
released alongside them `multabench-full-<name>`.

| What you are looking for | Where it is |
|--------------------------|-------------|
| Which datasets are in the benchmark | `multabench/datasets/all_multabench_datasets.py` |
| A dataset's Kaggle slug | `MulTaBenchDatasetID` in `multabench/datasets/all_datasets.py` |
| Where a dataset originally comes from | `multabench/datasets/sources.py` |
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
| Leaderboard | Elo via `bencheval`, RandomForest (default) anchored at 1000, missing results imputed with RandomForest and flagged, bootstrapped confidence intervals | Normalized scores per dataset | |
| Splits | Fixed and part of the task definition | Drawn when the run starts | |
| Group and time structure | Audited per dataset; group-aware or forward-in-time splits where needed | Not audited | |
| Runtime | Train and inference time per 1K rows | Same | ✅ |
| Hardware | Recorded per run | Recorded per run | ✅ |
| Outer splits | 3 folds, repeated 1–10 times depending on dataset size | 5 folds, no repeats | |
| Dataset size | Full size | Capped at 10K train and 2.5K test rows; a full-size mode is planned | |
| Inner validation | 8-fold bagging for every model | Each model holds out its own validation split | |
| Hyperparameters | Default, tuned, and tuned + ensembled | Default only | |
| Model implementations | TabArena's model registry | MulTaBench's own wrappers, which add image and text embeddings | |
