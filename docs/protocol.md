# Protocol

## Elo

Elo is computed with TabArena's own evaluation package, [bencheval](https://pypi.org/project/bencheval/),
pinned in the `leaderboard` extra. Every fold is its own battle: on each of a dataset's 6 folds, each pair of models
plays one battle, won by the lower test error on that fold (a tie counts half). The battles are fit as a Bradley–Terry
model, with RandomForest anchored at 1000. The 95% intervals come from 1,000 bootstrap resamples of the datasets, each
drawn with all 6 of its folds, so the intervals reflect how many datasets there are rather than how many folds. They
are bootstrapped before anchoring, so each model's interval is its own, RandomForest's included, rather than relative
to RandomForest. A model with no result on a fold has no battles there; nothing is imputed. Win rate, average rank, normalized
error and best-on count are per fold as well. Until October 2026 the leaderboard, like the paper, played one battle
per dataset on the mean error over its folds. That makes each dataset winner-take-all, which stretches the scale: on
the then 17 models, Kumo-Tabular was at 2670 rather than 2347, with the same ranking.

## Coverage

Default hyperparameters, `--size 10k`, 6 folds, frozen `e5-small` text and `dino-small` image embeddings
(ConTextTab and TabSTAR-v1.1 read raw text themselves; TabSTAR-v1.1's pretraining left MulTaBench out). TabPFN-v2,
v2.5 and Mitra-v2 take at most 10 classes, which leaves out 3 of the 80 datasets; TabFM and LimiX-2 cover them with
tabpfn-extensions' output coding. Mitra-v2 also runs out of GPU memory on MUL_TEXT_MELBOURNE_AIRBNB, whose 680
features it doesn't reduce since the table has categorical columns.

Every model that reads embeddings also ran with fine-tuned encoders: `e5-small-tar` on text datasets and
`dino-small-tar` on image datasets, LoRA-tuned on each fold's training rows (see [embeddings](embeddings.md)). Those
runs are in `multabench/leaderboard/models_tar/`, with the same failures. The leaderboard ranks each such model
as two entries, e.g. `LimiX-2 (Frozen)` and `LimiX-2 (TAR)`, and rates the encodings against each other.

## Reproducibility

Every model is seeded, and a rerun of the same run on the leaderboard's hardware gives the same test error, except for
four models whose packages aren't deterministic on GPU even with torch's deterministic algorithms. In a check of two
identical runs on six datasets, Mitra-v2 differed by up to 1.5%, TabSTAR-v1.1 by up to 0.25%, and Causilo
by up to 0.04%; TabDPT, TabICLv2 and Xiaomi-TabLDM differ by float rounding only (below 1e-7).

## MulTaBench and TabArena

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

## Reproducing the paper

The paper's code and results are at the tag [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version)
and the [`official-paper-version`](https://github.com/alanarazi7/MulTaBench/tree/official-paper-version) branch;
`master` has since moved to the leaderboard protocol.
