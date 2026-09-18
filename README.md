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

## From the Full text half

- [ ] **Name the fourth binned dataset.** The image curation appendix lists quantile binning for
      CS:GO Skins (10), PetFinder (8) and HubMAP HPA (10), explicitly as a non-exhaustive list.
      Either name the fourth or keep the list honest.

## Positioning and framing

- [ ] State plainly that MulTaBench is a **diagnostic benchmark for TAR, not a neutral ranking
      benchmark**: a true multimodal tabular architecture should excel on MulTaBench while
      remaining strong on simpler MMTL tasks. §6 gets close ("our objective is not to establish
      the SOTA") but only as an aside about selection bias.
- [ ] **Contextualize the effect size** for the +0.022 mean gain: TabArena's TFM-vs-XGBoost gaps
      are roughly +0.012 (default) and +0.007 (tuned), and TAR on a weak backbone is worth about
      as much as upgrading the backbone (RandomForest+TAR beats Frozen RealMLP).

## Concurrent work — VT-Bench

Not cited anywhere yet.

- [ ] **MUST ADDRESS: cite and differentiate VT-Bench** (`https://arxiv.org/pdf/2605.08146`,
      ICML 2026): a visual-tabular benchmark published after our submission, aggregating 14
      datasets across 9 domains and 756K samples. It is the closest concurrent work to the image
      half and the camera-ready cannot ignore it.
- [ ] **State the overlap honestly.** Seven of its eleven discriminative datasets are ones we
      also considered: Skin Cancer (PAD-UFES-20), DVM-Car, CelebA, PetFinder Adoption, Breast
      Cancer, Pawpularity and Anime (MyAnimeList). Two are already in MulTaBench, one is among
      the released extras, one we curated independently, and two we rejected as duplicates.
- [ ] **Draw the distinction on curation, not on size.** VT-Bench aggregates datasets and
      measures fusion; MulTaBench *screens* them, admitting only where the joint signal exceeds
      each unimodal baseline, and additionally requires task-awareness. A dataset an aggregating
      benchmark keeps is one we may reject for having no multimodal signal to measure — that is
      the diagnostic-versus-ranking argument, applied to dataset selection.
- [ ] **Mine its dataset table for candidates.** Its eleven discriminative datasets are the
      closest thing to a curated shortlist anyone has published for this problem, and four of
      them we have never evaluated. Worth working through before another blind Kaggle sweep.
- [ ] **Note what it has that we do not:** a cardiology/infarction set (44K) and three
      MIMIC-IV + MIMIC-CXR derived tasks. The MIMIC ones need PhysioNet credentialing, which is a
      redistribution constraint worth stating as a reason our benchmark is openly downloadable.

## Trimodal

- [ ] **Adopt the relaxed trimodal rule.** §4 and Appendix~E still report **2** trimodal datasets
      (PetFinder, Amazon Packages) under the strict rule, and the appendix only floats relaxation
      as future work. The scope decision is to report **8** and keep the strict pair as a stricter
      sub-tier. Verify all 8 pass Joint Signal on both modalities before claiming it.
- [ ] Answer veTL's framing question explicitly: we do **not** treat MMTL as two separate bimodal
      problems.
- [ ] **Reconcile the 9-vs-8 text-column mismatch**, still live. `tab:multabench_datasets` gives a
      non-zero text count to **9** image-tabular datasets; §4 and Appendix~E say **8**, and
      `FULLY_MULTIMODAL_DATASET_CANDIDATES` lists 8. The ninth is **HubMAP HPA**. Its only
      text-typed column is `rle`, the run-length encoded segmentation mask carried over from the
      source segmentation competition: strings of integer pairs, missing for most tiles. The
      semantic feature detector sees a high-cardinality object column and types it as text, so it
      reaches the table but was never a trimodal candidate. Two ways out, neither taken: qualify
      the prose (say 9 columns are typed as text, 8 of which are language), or drop `rle` in the
      HubMAP curation, which would change that dataset's feature counts and so needs a re-run.

## Small fixes found while mapping the paper

- [ ] The trimodal cross-reference in §3 points at `par:text_tabular_curation`; it should be
      `par:image_tabular_curation`.
- [ ] `\subsection{Computation Costs}` carries a `tab:costs` label; rename to `app:costs` (it is
      never referenced, and it collides conceptually with `tab:compute_costs`).
- [ ] `paper_production.py` regenerates tables whose captions have since drifted from the
      hand-edited `.tex`, so regenerating will clobber caption edits. Note also that
      `_get_datasets_table_latex()` reads the dataset table *back out of* `appendix.tex`, so that
      one table flows paper → script.

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
- [ ] Add STRABLE reference to the paper, as well as BeyondArena. Both should be considered
      concurrent work for text-tabular. Similarly, VT-Bench for images.
- [ ] Make a clear justification for the release of new datasets. Consider the phrasing such as
      "Datasets where Target-Aware Representations do not outperform Joint Frozen remain valuable,
      as they represent settings in which frozen encoders already provide sufficient task-relevant
      information and the main challenge lies in multimodal fusion rather than encoder adaptation."
      somewhere.
- [ ] Bimodal vs Trimodal (Weakness #2 + Question #1) --> emphasize more trimodality. Consider
      relaxing the writing / condition for it, and even make it more clear in the intro. "In
      hindsight, these criteria may have been overly restrictive; for instance, while our original
      setup required both text and images to show Task-Awareness gains, requiring Task-Awareness
      in just one unstructured modality would yield a broader trimodal collection. In fact, for
      all 8 of these datasets, both modalities pass the Joint Signal criteria, while the images
      alone satisfy the Task-Awareness criterion. Such relaxation could allow us to declare that
      MulTaBench already features 8 Trimodal datasets."
