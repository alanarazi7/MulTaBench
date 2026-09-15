# MulTaBench

Multimodal tabular benchmark with image and text modalities: 20 image-tabular and 20 text-tabular
datasets, with 40 further datasets released alongside them, 80 in total. Evaluates tabular
learners with optional DINO/E5 LoRA fine-tuning.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616)  
**Datasets**: [kaggle.com/chico89](https://www.kaggle.com/chico89/datasets)

## Setup

```bash
source init.sh           # installs Python 3.11, creates .venv, installs deps via uv
source .venv/bin/activate
cp .env.example .env     # WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
```

## Running the benchmark

```bash
python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --multimodal_state all
```

`--model`: `light` (LightGBM), `cat`, `xgb`, `rf`, `realmlp`, `tabm`, `tabicl`, `tabdpt`,
`tabpfnv2`, `tabstar`, `autogluon`, `contexttab`. Append `_opt` for the tuned variant.

`--multimodal_state` picks which features are active: `all` (tabular + image + text), `non`
(tabular + text), `img`, `txt`, `no_img`, `no_txt`, `ft` (tabular + fine-tuned encoders), `all 🔥`
(all features + fine-tuned encoders). Fine-tuning is configured with `--tune_dino yes --dino_lr
--dino_rank --dino_img_layers` and the matching `--tune_e5` flags.

## Datasets

Downloaded automatically via `kagglehub` from the `chico89` Kaggle account. Core datasets use the
slug `multabench-<name>`, the 40 released alongside use `multabench-full-<name>`; a dataset's
exact slug is its `SLUG_BASE` in `multabench/benchmark/datasets/<DATASET_ID>.py`.

| What | Where |
|------|-------|
| Dataset registry (`MulTaBenchDatasetID`, `{TASK}_{MODALITY}_{NAME}`) | `multabench/datasets/all_datasets.py` |
| Membership lists, 20 each: `MULTABENCH_CORE_IMAGE`, `MULTABENCH_CORE_TEXT`, `MULTABENCH_FULL_IMAGE_EXTRA`, `MULTABENCH_FULL_TEXT_EXTRA` | `multabench/datasets/all_multabench_datasets.py` |
| Curation recipe per dataset (target, features, loading) | `multabench/datasets/annotated/` |
| Per-dataset properties (rows, classes, feature counts) | `multabench/leaderboard/results/datasets_summary{,_extra}.csv` |

The two `*_EXTRA` lists are the datasets released alongside the benchmark. They are admitted on
*Joint Signal* alone; the *Task-awareness* condition was never run on them, so their scores must
not be pooled with the 40 MulTaBench datasets, and they carry no tier name of their own. Four of
them ship a quantile-binned target, so a `BIN_`/`MUL_` name can have a `REG_` source.

## Architecture

- **Image encoder**: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S, 384-dim CLS token), optional LoRA on the last N attention layers
- **Text encoder**: `intfloat/e5-small-v2` (384-dim mean pool), columns formatted as `"passage: col_name: col_value"`
- **PCA**: both encoders reduced to 30 components by default
- **Splits**: 90/10 train/test (stratified for classification), max 2000 test examples

Code lives under `multabench/`: `datasets/` and `benchmark/` (loading, curation, Kaggle upload),
`dino/` `e5/` `finetune/` (encoders and LoRA), `preprocessing/`, `baselines/` (models and
evaluation), `leaderboard/` (Streamlit dashboard and result CSVs), `scripts/` (`do_*.py`
utilities, entry point `do_leaderboard.py`), `utils/`.

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

## Track 2 — Analyses (consolidate what exists; fill the two real gaps)

Most of this already exists; the job is to get it into one place, make it runnable, and commit
its outputs. Two items are genuine gaps.

- [ ] **Consolidate all rebuttal analysis code onto `master`.** Still split: `elo_leaderboard.py`
      only on `elo-leaderboard`; the ρ-sweep (`threshold_grid.py`, `curation_accept.py`,
      `delta_sweep.py`) and 22 result CSVs only on `neurips-rebuttal-sensitivity`;
      `model_agreement.py` only on `master`; the paired t-test and the TabArena comparison only
      in the private `internal-MulTaBench`. **No single checkout reproduces the rebuttal**, and
      five analysis CSVs on `master` have no generating code here.
- [ ] **Fix `model_agreement.py`** — it fails on import as committed (`build_pass_matrix` moved
      from `committee_pool.py` to `pass_matrix.py`), and it persists no CSV. Commit the agreement
      matrix as a CSV alongside the two currently-untracked PNGs.
- [ ] **GAP — the "of 40" δ/ρ sensitivity numbers have no generating code.** Everything committed
      is denominated in the 56-dataset *text* pool, but the rebuttal quotes 32/40, 30/40, 34/40,
      30/40 and 20/40 over the full benchmark. The image-side sweep does not exist
      (`committee_pool.py` is hard-wired to `no_text`/`text_only`). Build the image pool CSV and
      pass matrix, then re-derive the combined numbers — **verify the rebuttal figures reproduce
      before they go into the paper.**
- [ ] **GAP — no Benjamini–Hochberg correction and no artifact for the significance test.** The
      existing script runs an uncorrected one-sided `ttest_1samp` and only prints. Add the BH-FDR
      step, write a committed CSV, and confirm the claimed 37/40 significant with 3 exceptions
      (2 image-tabular, 1 text-tabular).
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

- [ ] **Move the δ/ρ threshold discussion from Appendix A into the main text** — explicitly
      promised to both 2eKq and jcEc.
- [ ] Add the δ and ρ sensitivity results, framing the need for TAR as a **spectrum, not a strict
      binary condition**.
- [ ] Summarize the **committee simulation** (C(10,5) = 252 panels) and the **pairwise model
      agreement** result showing the two TabPFN variants are not a voting bloc (78% agreement,
      identical to RandomForest↔TabPFN-2.5; average 70%, range 59–82%).
- [ ] One-sentence mention of the **per-dataset significance** result (37/40), pointing to the
      appendix.

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
- [ ] Per-dataset paired t-test with BH-FDR, naming the 3 non-significant datasets.
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
      somewhere. I think it should appear in the results.
- [ ] For Table 9, how many image-tabular datasets have an individually significant TAR>Frozen
      gain (per-dataset CIs)? Given the +0.022 mean? ---> same, significance should appear in main
      results+appendix.
- [ ] ELO Rating (Question #3) --> add Elos to appendix.
- [ ] Bimodal vs Trimodal (Weakness #2 + Question #1) --> emphasize more trimodality. Consider
      relaxing the writing / condition for it, and even make it more clear in the intro. "In
      hindsight, these criteria may have been overly restrictive; for instance, while our original
      setup required both text and images to show Task-Awareness gains, requiring Task-Awareness
      in just one unstructured modality would yield a broader trimodal collection. In fact, for
      all 8 of these datasets, both modalities pass the Joint Signal criteria, while the images
      alone satisfy the Task-Awareness criterion. Such relaxation could allow us to declare that
      MulTaBench already features 8 Trimodal datasets."
