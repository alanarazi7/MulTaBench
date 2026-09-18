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

## Positioning and framing

- [ ] State plainly that MulTaBench is a **diagnostic benchmark for TAR, not a neutral ranking
      benchmark**: a true multimodal tabular architecture should excel on MulTaBench while
      remaining strong on simpler MMTL tasks. §6 gets close ("our objective is not to establish
      the SOTA") but only as an aside about selection bias.
- [ ] **Contextualize the effect size** for the +0.022 mean gain: TabArena's TFM-vs-XGBoost gaps
      are roughly +0.012 (default) and +0.007 (tuned), and TAR on a weak backbone is worth about
      as much as upgrading the backbone (RandomForest+TAR beats Frozen RealMLP).

## Small fixes found while mapping the paper

- [x] `paper_production.py` captions and headers now match the hand-edited `.tex`, so regenerating
      no longer clobbers the paper. The second half of this note was wrong:
      `_get_datasets_table_latex()` builds from `datasets_summary.csv`, not from `appendix.tex`.
      Nothing under `multabench/leaderboard/` reads a `.tex` file, so every table flows
      script → paper.
- [x] `tab:text_deduplication` is generated rather than hand-written. It reproduces the table's 15
      rows and their check marks, and independently confirms the 56-candidate pool.
- [ ] `tab:text_curation_grid` still differs in layout: the generator emits `{lcccccc}` with plain
      `\midrule` group separators, the paper `{llcccccr}` with `Approved`/`Rejected` group
      headings. Aligning them changes the typeset table, so it is deliberately left alone.

---

# Alan's Manual TODOs

- [ ] Clarify we are maybe not a pure benchmark: "We would like to emphasize its purpose:
      MulTaBench is not intended as a neutral ranking benchmark for arbitrary multimodal tabular
      models, but is better framed as a diagnostic benchmark for studying tasks where fusion of
      modality-based representations is required and target-aware unstructured representations
      are necessary." This could be intro/discussion material. "Our work establishes the need for
      Target-Aware Representations (TAR), providing a dedicated benchmark to evaluate solutions
      for it. A true multimodal tabular architecture should excel on MulTaBench while remaining
      strong on simpler MMTL tasks." -> supports the same, and thus we also release extended.
- [ ] Make a clear justification for the release of new datasets. Consider the phrasing such as
      "Datasets where Target-Aware Representations do not outperform Joint Frozen remain valuable,
      as they represent settings in which frozen encoders already provide sufficient task-relevant
      information and the main challenge lies in multimodal fusion rather than encoder adaptation."
      somewhere.
