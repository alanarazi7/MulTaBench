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

# Camera Ready TODOs (NeurIPS 2026 D&B)

Commitments made during the rebuttal (reviewers 2eKq, jcEc, veTL; AC douQ) that must land in
the camera-ready version. Source of truth: the OpenReview discussion thread for submission 240.

**Working agreement.** All camera-ready work happens on a feature branch of this repo, committed
and pushed as we go; branches are reviewed and merged to `master` by the maintainer. Paper edits
live in the separate `paper-multabench` repo.

**Scope decisions.** The release is 80 datasets, but only 40 of them are MulTaBench. The other 40
are admitted on *Joint Signal* alone and carry no tier name: no `MulTaBench-Core`, no
`MulTaBench-Full`. MulTaBench keeps its published meaning, the 40 curated datasets, and every
analysis stays denominated in those 40, since the *Joint TAR* condition was never run on the
other 40 and the two must not be pooled. The camera-ready gets one extra page (10 total): a new
main-text section carries the δ/ρ sensitivity and the repositioning, while Elo and the
per-dataset significance tables go to the appendix. The relaxed trimodal criterion is adopted.
Sequencing is datasets-first.

## Track 1 — Datasets (done, except the trimodal extension)

Both halves are closed: 40 MulTaBench datasets plus 40 released alongside them, 80 in total,
image 40 (20 CLS / 20 REG) and text 40 (20 CLS / 20 REG). The 40 released alongside are now
documented end to end, which was the longest-lead item on this list:

- `datasets_summary.csv` and `datasets_summary_extra.csv` are regenerated (#43), with date
  columns kept out of the text features, so the reported text counts match what the pipeline
  actually encodes.
- `paper_production.py` emits all three appendix property tables (core, additional joint signal,
  additional properties) rather than leaving them hand-maintained (#39, #43).
- All 80 datasets have a per-dataset description in the paper appendix, each a high-level
  summary with no row or feature counts (`paper-multabench` #4 and #6; #7 is open for the
  regenerated core table).

One item remains:

- [ ] **Extend the trimodal group toward ~15** (the rebuttal estimate for Full) by detecting text
      columns on the new image-tabular datasets.

## Track 2 — Analyses (consolidate what exists; fill the remaining gap)

Most of this already exists; the job is to get it into one place, make it runnable, and commit
its outputs. One item is a genuine gap.

- [ ] **Consolidate all rebuttal analysis code onto `master`.** The ρ-sweep
      (`threshold_grid.py`, `curation_accept.py`, `delta_sweep.py`) and
      `benchmark_threshold_sweep.py` have landed. Still split: `elo_leaderboard.py` only on
      `elo-leaderboard`, and the TabArena comparison only in the private `internal-MulTaBench`.
      **No single checkout reproduces the rebuttal**, and five analysis CSVs on `master` have no
      generating code here.
- [ ] **Fix `model_agreement.py`** — it fails on import as committed (`build_pass_matrix` moved
      from `committee_pool.py` to `pass_matrix.py`), and it persists no CSV. Commit the agreement
      matrix as a CSV alongside the two currently-untracked PNGs.
- [ ] **Add the four committee-consensus bucket counts to code** (full consensus 33/56, near
      consensus 45/56, strong majority 52/56, borderline 4/56). They reproduce from
      `committee_delta_sweep.csv`, but no script prints them — the framing currently exists only
      in rebuttal prose.
- [ ] **Elo — already complete** (`elo_frozen_vs_tar.csv`, 27 competitors, RandomForest Frozen
      anchored at 1000). Fix the stale "23 competitors" docstring and merge.

## Track 3 — Release

- [ ] Merge the consolidated analysis branch so every rebuttal number is reproducible from a
      single checkout of `master`.
- [ ] Switch the paper's code URL from the anonymous repo to this one.

## Protected branches

Two branches hold work that exists nowhere else. **Do not delete them** until their content is
merged or explicitly abandoned:

| branch | what only lives there |
|--------|-----------------------|
| `neurips-rebuttal-sensitivity` | the rebuttal sensitivity analyses and 22 result CSVs |
| `elo-leaderboard` | `elo_leaderboard.py` and the two Elo CSVs |

---

# Paper TODOs (`paper-multabench` repo)

The companion-release framing has landed on `main` (abstract, §4, §8, checklist) and the
appendix section plus its admission table are on the `core-full-framing` branch. Everything
below is what remains: a paper-side consequence of work that has landed in this repo, or a
commitment from the rebuttal.

Main-text body is currently over the 10-page camera-ready limit, so every addition needs a
matching trim. Switch `neurips_2026.tex` from `[preprint]` to `[eandd, final]` and compile early
to get a real page count.

## From the Full text half

- [ ] **Say that admission was measured on the *uploaded* artifacts**, not on the pool's original
      sources. The δ=0.001 / ρ=3-of-5 rule is already stated in the new appendix section; this
      qualifier is not.
- [ ] **Describe the four binned datasets** in the curation appendix: their targets are cut into
      equal-frequency quantile bins at curation time, so a `BIN_`/`MUL_` dataset can have a `REG_`
      source. Name them and give their bin counts.
- [ ] **Say that membership was decided on the uploaded artifacts, not the pool ranking.** Two
      datasets differ from the pool-ranked draft: IMDB Genre and Melbourne Airbnb are in, Movies
      Revenue and ML/DS/AI Salaries are out.
- [ ] **Carry the California Prices caveat.** It is admitted on all five models but at +0.002 on
      each — 2× δ at the reported 3-decimal precision. Read the sign, not the ranking.
- [ ] **Note the non-unanimous admissions**: five of the 20 pass 4 of 5 rather than 5 of 5.
- [ ] **Record the two curation deviations** that change what a reader would compute from the
      source: Consumer Complaint is capped at 100K rows, and Melbourne Airbnb drops its URL
      columns (they would otherwise be detected as image features).

## New main-text section (Curation Robustness)

- [ ] **Still missing: the pairwise model agreement** result showing the two TabPFN variants are
      not a voting bloc (78% agreement, identical to RandomForest↔TabPFN-2.5; average 70%, range
      59–82%). Nothing in the paper carries it yet, and `model_agreement.py` persists no CSV.

## Positioning and framing

- [ ] State plainly that MulTaBench is a **diagnostic benchmark for TAR, not a neutral ranking
      benchmark**: a true multimodal tabular architecture should excel on MulTaBench while
      remaining strong on simpler MMTL tasks.
- [ ] Frame MulTaBench as a **living benchmark** (TabArena analogy) built on an open pipeline.
- [ ] **Strengthen the novelty framing** (veTL): 15 genuinely new image-tabular datasets, screened
      from 1000+ Kaggle datasets down to a 100+ candidate pool; surface the Appendix D
      engineering work (corrupt images, task formulation, directory standardization) into the
      narrative; present the unified API as the reusable community contribution.
- [ ] **Contextualize the effect size** for the +0.022 mean gain: TabArena's TFM-vs-XGBoost gaps
      are roughly +0.012 (default) and +0.007 (tuned), and TAR on a weak backbone is worth about
      as much as upgrading the backbone (RandomForest+TAR beats Frozen RealMLP).

## Concurrent work — VT-Bench

- [ ] **MUST ADDRESS: cite and differentiate VT-Bench** (`https://arxiv.org/pdf/2605.08146`,
      ICML 2026): a visual-tabular benchmark published after our submission, aggregating 14
      datasets across 9 domains and 756K samples. It is the closest concurrent work to the image
      half and the camera-ready cannot ignore it.
- [ ] **State the overlap honestly.** Seven of its eleven discriminative datasets are ones we
      also considered: Skin Cancer (PAD-UFES-20), DVM-Car, CelebA, PetFinder Adoption, Breast
      Cancer, Pawpularity and Anime (MyAnimeList). Two are already in Core, one is admitted to
      Full, one we curated independently, and two we rejected as duplicates of Core entries.
- [ ] **Draw the distinction on curation, not on size.** VT-Bench aggregates datasets and
      measures fusion; MulTaBench *screens* them, admitting only where the joint signal exceeds
      each unimodal baseline, and Core additionally requires task-awareness. A dataset an
      aggregating benchmark keeps is one we may reject for having no multimodal signal to
      measure — that is the diagnostic-versus-ranking argument, applied to dataset selection.
- [ ] **Mine its dataset table for candidates.** Its eleven discriminative datasets are the
      closest thing to a curated shortlist anyone has published for this problem, and four of
      them we have never evaluated. Worth working through before another blind Kaggle sweep.
- [ ] **Note what it has that we do not:** a cardiology/infarction set (44K) and three
      MIMIC-IV + MIMIC-CXR derived tasks. The MIMIC ones need PhysioNet credentialing, which is a
      redistribution constraint worth stating as a reason our benchmark is openly downloadable.

## Core/Full and trimodal

- [ ] **Adopt the relaxed trimodal rule** in §4 and Appendix E: report **8 trimodal datasets**,
      keeping the strict-rule result (PetFinder, Amazon Packages) as a stricter sub-tier. Verify
      all 8 pass Joint Signal on both modalities before claiming it. Note that Full is expected
      to reach ~15.
- [ ] Answer veTL's framing question explicitly: we do **not** treat MMTL as two separate bimodal
      problems.
- [ ] **Reconcile the 9-vs-8 text-column mismatch.** Table 3 in the appendix counts **9**
      image-tabular datasets with a text feature; §4 and Appendix E say **8**, and
      `FULLY_MULTIMODAL_DATASET_CANDIDATES` lists 8. The ninth is **HubMAP HPA**. Its only
      text-typed column is `rle`, the run-length encoded segmentation mask carried over from the
      source segmentation competition: strings of integer pairs, missing for most tiles. The
      semantic feature detector sees a high-cardinality object column and types it as text, so it
      reaches the table but was never a trimodal candidate. Two ways out, neither taken yet:
      qualify the prose (say 9 columns are typed as text, 8 of which are language), or drop `rle`
      in the HubMAP curation, which would change that dataset's feature counts and so needs a
      re-run. **Nothing about this is in the paper yet** — the paper still says 8 with no
      explanation of the ninth.

## New appendix material

- [ ] Elo / Bradley–Terry leaderboard table (27 competitors) plus a method description.
- [ ] Pairwise model agreement matrix.
- [ ] Committee simulation detail and the ρ / δ sweep tables.
- [ ] The image-tabular rejected pool, as a curation record.

## Small fixes found while mapping the paper

- [ ] The trimodal cross-reference in §3 points at `par:text_tabular_curation`; it should be
      `par:image_tabular_curation`.
- [ ] `\subsection{Computation Costs}` carries a `tab:costs` label; rename to `app:costs` (it is
      never referenced, and it collides conceptually with `tab:compute_costs`).
- [ ] `checklist.tex` hardcodes "Section 7" for limitations; this goes stale once a section is added.
- [ ] `paper_production.py` regenerates tables whose captions have since drifted from the
      hand-edited `.tex`, so regenerating will clobber caption edits. Note also that
      `_get_datasets_table_latex()` reads the dataset table *back out of* `appendix.tex`, so that
      one table flows paper → script.

## Known defect on `main`

- [ ] `checklist.tex` new-assets answer begins `Justification: Justification:`. Fix on the next
      paper branch.

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
- [ ] Add analysis of voting correlations between pairs of learners. [Reviewer 1, Weakness #2 +
      Question #2]
- [ ] Add committee analysis - "committee simulation" - % acceptance had we used 5 different
      models.
- [ ] "The acceptance threshold appears somewhat arbitrary" -> show robustness analysis to
      Performance margin threshold and to Voting Consensus Threshold
- [ ] Effect Size (Weakness #1 + Question #1) -> consider mentioning the effect size analysis
      somewhere. I think it should appear in the results. Note: Cohen's d was deliberately dropped
      from both the paper and the CSV (#48), so this is still open.
- [ ] ELO Rating (Question #3) --> add Elos to appendix.
- [ ] Bimodal vs Trimodal (Weakness #2 + Question #1) --> emphasize more trimodality. Consider
      relaxing the writing / condition for it, and even make it more clear in the intro. "In
      hindsight, these criteria may have been overly restrictive; for instance, while our original
      setup required both text and images to show Task-Awareness gains, requiring Task-Awareness
      in just one unstructured modality would yield a broader trimodal collection. In fact, for
      all 8 of these datasets, both modalities pass the Joint Signal criteria, while the images
      alone satisfy the Task-Awareness criterion. Such relaxation could allow us to declare that
      MulTaBench already features 8 Trimodal datasets."
