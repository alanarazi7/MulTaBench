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

# Protected branches

**Do not delete these**, even though `master` now carries the analyses they were opened for:

| branch | what only lives there |
|--------|-----------------------|
| `neurips-rebuttal-sensitivity` | `model_sensitivity.py` and its `model_*.csv` outputs: a second computation of the committee results, dropped in favour of `committee_sensitivity.py`, which it agreed with exactly |
| `elo-leaderboard` | `elo_leaderboard.py` and its three Elo CSVs |

- [ ] Merge `elo-leaderboard`, the last analysis branch whose numbers are not reproducible from a
      single checkout of `master`.

The other analysis branches are squash-merged, so `git branch --no-merged` still lists them;
compare contents, not the merge flag, before deleting any.

---

# Paper TODOs (`paper-multabench` repo)

Main-text body is currently over the 10-page camera-ready limit, so every addition needs a
matching trim. Switch `neurips_2026.tex` from `[preprint]` to `[eandd, final]` and compile early
to get a real page count.

## From the Full text half

The admission table (`tab:extra_datasets`) now reports each dataset's margin and its pass count,
so the numbers behind these are printed; what is missing is the prose that tells a reader how to
read them.

- [ ] **Say that admission was measured on the *uploaded* artifacts**, not on the pool's original
      sources. The δ=0.001 / ρ=3-of-5 rule is stated in the curation appendix; this qualifier is
      not.
- [ ] **Say that membership was decided on the uploaded artifacts, not the pool ranking.** Two
      datasets differ from the pool-ranked draft: IMDB Genre and Melbourne Airbnb are in, Movies
      Revenue and ML/DS/AI Salaries are out. As it stands the pool table rejects both of the
      first two with no word on why they are released.
- [ ] **Carry the California Prices caveat.** The table shows +0.002 on 5 of 5, which is 2× δ at
      the reported 3-decimal precision. Read the sign, not the ranking.
- [ ] **Record the two curation deviations** that change what a reader would compute from the
      source: Consumer Complaints is capped at 100K rows (the properties table prints 100,000
      with no note), and Melbourne Airbnb drops its URL columns (they would otherwise be detected
      as image features).
- [ ] **Name the fourth binned dataset.** The image curation appendix lists quantile binning for
      CS:GO Skins (10), PetFinder (8) and HubMAP HPA (10), explicitly as a non-exhaustive list.
      Either name the fourth or keep the list honest.

## Positioning and framing

- [ ] State plainly that MulTaBench is a **diagnostic benchmark for TAR, not a neutral ranking
      benchmark**: a true multimodal tabular architecture should excel on MulTaBench while
      remaining strong on simpler MMTL tasks. §6 gets close ("our objective is not to establish
      the SOTA") but only as an aside about selection bias.
- [ ] Frame MulTaBench as a **living benchmark** (TabArena analogy) built on an open pipeline.
- [ ] **Strengthen the novelty framing** (veTL): 15 genuinely new image-tabular datasets, screened
      from 1000+ Kaggle datasets down to a 100+ candidate pool; surface the image-curation
      engineering work (corrupt images, task formulation, directory standardization) into the
      narrative; present the unified API as the reusable community contribution.
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

## Elo

The appendix subsection and its table are written but commented out in `appendix.tex` (the block
above `\section{MulTaBench Datasets}`), because the numbers are not reproducible from `master`.

- [ ] Bring `elo_leaderboard.py` and its CSVs over from the `elo-leaderboard` branch.
- [ ] Uncomment the appendix subsection once they are on `master`, and check the 27-competitor
      table against the regenerated numbers.

## Small fixes found while mapping the paper

- [ ] The trimodal cross-reference in §3 points at `par:text_tabular_curation`; it should be
      `par:image_tabular_curation`.
- [ ] `\subsection{Computation Costs}` carries a `tab:costs` label; rename to `app:costs` (it is
      never referenced, and it collides conceptually with `tab:compute_costs`).
- [ ] `paper_production.py` regenerates tables whose captions have since drifted from the
      hand-edited `.tex`, so regenerating will clobber caption edits. Note also that
      `_get_datasets_table_latex()` reads the dataset table *back out of* `appendix.tex`, so that
      one table flows paper → script.

## Settled, no action

- **Curation robustness.** Committee simulation, per-candidate pass rates and pairwise agreement
  are all in the paper. Agreement is reported raw, not as Cohen's κ, which was dropped from the
  analysis in #54.
- **Pairwise agreement matrix.** In the appendix as `figures/model_agreement.pdf`.
- **Non-unanimous admissions.** The admission table's Pass column carries them per dataset, which
  is stronger than the "five of 20 pass 4 of 5" sentence we had planned.
- **The image-tabular rejected pool** is deliberately not reported at text-tabular detail: the
  appendix states we could not assure faithful curation of the failures, and says so.
- **The curation-robustness figure** was cut from the paper, and its generator deleted here (#56).

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
- [ ] Effect Size (Weakness #1 + Question #1) -> consider mentioning the effect size analysis
      somewhere. I think it should appear in the results. Note: Cohen's d was deliberately dropped
      from both the paper and the CSV (#48), so this is still open.
- [ ] ELO Rating (Question #3) --> add Elos to appendix. `elo_leaderboard.py` and its three CSVs
      are still only on the `elo-leaderboard` branch; bring them over in their own PR. The
      subsection and table are drafted in `appendix.tex` but stay commented out until the code
      lands on `master`.
- [ ] Bimodal vs Trimodal (Weakness #2 + Question #1) --> emphasize more trimodality. Consider
      relaxing the writing / condition for it, and even make it more clear in the intro. "In
      hindsight, these criteria may have been overly restrictive; for instance, while our original
      setup required both text and images to show Task-Awareness gains, requiring Task-Awareness
      in just one unstructured modality would yield a broader trimodal collection. In fact, for
      all 8 of these datasets, both modalities pass the Joint Signal criteria, while the images
      alone satisfy the Task-Awareness criterion. Such relaxation could allow us to declare that
      MulTaBench already features 8 Trimodal datasets."
