# MulTaBench

A benchmark for multimodal tabular learning: tables whose columns include images or free text, not
just numbers and categories. It is 20 image-tabular and 20 text-tabular curated datasets, with 40
further datasets released alongside them, 80 in total. The benchmark evaluates tabular learners
under a target-aware setting, where the image and text encoders are fine-tuned on the task rather
than used frozen.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616)  
**Datasets**: [kaggle.com/chico89](https://www.kaggle.com/chico89/datasets)

## Getting started

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env     # Weights & Biases, Hugging Face and Kaggle credentials

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --multimodal_state all
```

`benchmark.py` is the single entry point: it evaluates one model on one dataset and logs the
result to Weights & Biases. `--help` lists the available models and the feature combinations each
run can use, from tabular-only through fully multimodal with fine-tuned encoders.

## Datasets

Every dataset is hosted on the [`chico89`](https://www.kaggle.com/chico89/datasets) Kaggle account
and downloaded on demand; the benchmark datasets are slugged `multabench-<name>` and the ones
released alongside them `multabench-full-<name>`.

| What you are looking for | Where it is |
|--------------------------|-------------|
| Which datasets are in the benchmark | `multabench/datasets/all_multabench_datasets.py` |
| A dataset's Kaggle slug and source | `multabench/benchmark/datasets/` |
| How a dataset was curated | `multabench/datasets/annotated/` |
| Size, task and feature counts per dataset | `multabench/leaderboard/results/datasets_summary{,_extra}.csv` |

The 40 datasets released alongside the benchmark were admitted on a weaker criterion and carry no
tier name of their own, so their scores must not be pooled with the 40 MulTaBench datasets.

---

# Paper TODOs (`paper-multabench` repo)

Main-text body is currently over the 10-page camera-ready limit, so every addition needs a
matching trim. Switch `neurips_2026.tex` from `[preprint]` to `[eandd, final]` and compile early
to get a real page count.

- [ ] **Contextualize the effect size** for the +0.022 mean gain: TabArena's TFM-vs-XGBoost gaps
      are roughly +0.012 (default) and +0.007 (tuned), and TAR on a weak backbone is worth about
      as much as upgrading the backbone (RandomForest+TAR beats Frozen RealMLP).
