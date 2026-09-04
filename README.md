# MulTaBench

Multimodal tabular benchmark with image and text modalities. Evaluates tabular learners on 20 image datasets and 20 text datasets, with optional DINO/E5 LoRA fine-tuning.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616)  
**Datasets**: [kaggle.com/chico89](https://www.kaggle.com/chico89/datasets)

## Setup

```bash
source init.sh           # installs Python 3.11, creates .venv, installs deps via uv
source .venv/bin/activate
cp .env.example .env     # fill in your credentials
```

Credentials (`.env`):
```
WANDB_API_KEY=...
WANDB_ENTITY=...
HF_TOKEN=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

## Running the Benchmark

```bash
python benchmark.py \
    --model light \
    --dataset_name MUL_IMAGE_PETFINDER \
    --fold 0 \
    --multimodal_state "all"
```

With LoRA fine-tuning:
```bash
python benchmark.py \
    --model tabm \
    --dataset_name MUL_IMAGE_PETFINDER \
    --fold 0 \
    --multimodal_state "all 🔥" \
    --tune_dino yes --dino_lr 0.001 --dino_rank 16 --dino_img_layers 3 \
    --tune_e5 yes --e5_lr 1e-4 --e5_rank 16 --e5_text_layers 3
```

### `--model` options

| Key | Model |
|-----|-------|
| `light` | LightGBM |
| `cat` | CatBoost |
| `xgb` | XGBoost |
| `rf` | Random Forest |
| `realmlp` | RealMLP |
| `tabm` | TabM |
| `tabicl` | TabICL v2 |
| `tabdpt` | TabDPT |
| `tabpfnv2` | TabPFN v2 |
| `tabstar` | TabSTAR |
| `autogluon` | AutoGluon Multimodal |
| `contexttab` | ConTextTab |

Append `_opt` for hyperparameter-optimized variants (e.g. `light_opt`).

### `--multimodal_state` options

| Value | Features used |
|-------|---------------|
| `all` | tabular + image + text |
| `non` | tabular + text only |
| `img` | image only |
| `txt` | text only |
| `no_img` | tabular only (no image) |
| `no_txt` | tabular + image only |
| `all 🔥` | all features + fine-tuned encoders |
| `ft` | tabular + fine-tuned image + fine-tuned text |

## Datasets

60 datasets hosted on Kaggle under `multabench-*`, downloaded automatically via `kagglehub`.
Names follow `{TASK}_{MODALITY}_{NAME}` where task is `BIN`/`MUL`/`REG`. The registry is
`MulTaBenchDatasetID` (`multabench/datasets/all_datasets.py`); the benchmark lists live in
`multabench/datasets/all_multabench_datasets.py`.

**`MULTABENCH_CORE_IMAGE`** (20): celebrity attractiveness, hateful memes, mammography, CheXpert, CBIS-DDSM, glaucoma, CS:GO skins, flower bouquets, HuBMAP, Instagram engagement, PetFinder adoption, zooplankton, Amazon bestsellers, Amazon packages, H&M fashion, Khaadi clothes, Letterboxd movies, mango mass, photography bots, painting price.

**`MULTABENCH_CORE_TEXT`** (20): fake job postings, Jigsaw toxicity, Kickstarter, data scientist salary, Michelin guide, product sentiment, Spotify genres, US accidents, wine reviews, women's clothing, baby products, book price, book readability, Mercari, Montgomery salaries, Rotten Tomatoes, SciMago, Vancouver salaries, video game sales, Zomato.

**`MULTABENCH_FULL_TEXT_EXTRA`** (20): the Full-tier text additions, uploaded as `multabench-full-*`. Four ship a quantile-binned target, which is why their names are `BIN_`/`MUL_` while their sources are `REG_`.

## Architecture

```
multabench/
  datasets/       # dataset loading, curation, all_datasets enum
  dino/           # DINO ViT image encoder + LoRA fine-tuning
  e5/             # E5 text encoder + LoRA fine-tuning
  preprocessing/  # feature detection, splits, PCA projection
  finetune/       # training args for DINO/E5 fine-tuning
  baselines/      # all model implementations + evaluation
  benchmark/      # MulTaBench dataset loading (Kaggle-hosted)
  leaderboard/    # Streamlit dashboard + result CSVs
  scripts/        # standalone utility and figure scripts
  utils/          # logging, I/O, metrics
```

- **Image encoder**: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S, 384-dim CLS token), optional LoRA on last N attention layers
- **Text encoder**: `intfloat/e5-small-v2` (384-dim mean pool), columns formatted as `"passage: col_name: col_value"`
- **PCA**: both encoders reduced to 30 components by default
- **Splits**: 90/10 train/test (stratified for classification), max 2000 test examples

## Scripts (`multabench/scripts/`)

| Script | Purpose |
|--------|---------|
| `do_leaderboard.py` | Streamlit leaderboard dashboard |
| `do_finetune_save.py` | Fine-tune and save DINO checkpoint |
| `do_attention.py` | DINO attention map visualization |
| `do_tagging.py` | Interactive dataset annotation tool |
| `do_kaggle_prepare.py` | Prepare dataset for Kaggle upload |
| `do_kaggle_upload.py` | Upload curated dataset to Kaggle |
| `do_multabench_audit.py` | Validate all benchmark datasets |
| `do_dataset_summary.py` | Dataset statistics summary |
| `do_paper.py` | Paper figure production |

---

# Camera Ready TODOs (NeurIPS 2026 D&B)

Commitments made during the rebuttal (reviewers 2eKq, jcEc, veTL; AC douQ) that must land in
the camera-ready version. Source of truth: the OpenReview discussion thread for submission 240.

**Working agreement.** All camera-ready work happens on a feature branch of this repo, committed
and pushed as we go; branches are reviewed and merged to `master` by the maintainer. Paper edits
live in the separate `paper-multabench` repo.

**Scope decisions.** MulTaBench-Full is the full 80 datasets as promised. The camera-ready gets
one extra page (10 total): a new main-text section carries the δ/ρ sensitivity and the
repositioning, while Elo and the per-dataset significance tables go to the appendix. The relaxed
trimodal criterion is adopted. Sequencing is datasets-first.

## Track 1 — Datasets (start first; longest lead time)

Blocking track. The text half is closed; the image side is the real work.

- [x] **Name the benchmark lists in the registry.** `MULTABENCH_CORE_IMAGE`, `MULTABENCH_CORE_TEXT`
      and `MULTABENCH_FULL_TEXT_EXTRA` live in `multabench/datasets/all_multabench_datasets.py`
      (#18, #21). No `Tier` enum or accessor API: the lists carry the membership.
- [x] **Text-tabular Full (20 → 40).** The 20 extras are curated, uploaded as `multabench-full-*`,
      registered, and verified on their uploaded artifacts — 20 of 20 admitted on Joint Signal
      (#20, #21). Four ship a quantile-binned target. Reproduce with
      `python -m multabench.leaderboard.analysis.text_full_verification`.
- [ ] **Image-tabular Full (20 → 40).** The big lift: ~20 additional image-tabular datasets
      passing Joint Signal. Sources: the ~13 non-published image entries that already have
      `annotated/` curation modules, the BagOfTricks set, and fresh Kaggle curation. Each
      needs curate → validate → upload → evaluate.
- [ ] **Record the image-tabular rejected pool.** There is no image analogue of
      `REJECTED_TEXT_DATASETS`, and no committed results CSVs for any rejected image candidate.
      Needed both for the Full tier and to close the symmetry gap in the curation appendix.
- [ ] **Run curation evaluation on every new image candidate** (4 conditions × 5 curation learners
      × 5 folds) so Joint Signal is decided on evidence. Sweeps created locally; single-agent
      warmup per newly uploaded dataset before fanning out.
- [ ] **Upload all new image datasets to Kaggle** under the unified API (`multabench-<name>`
      slugs under `chico89`; flat `images/` + `data.csv` + `metadata.json`), validating with
      `compare_df_summaries()` first and dropping truncated or corrupt images.
- [ ] **Extend the trimodal group toward ~15** (the rebuttal estimate for Full) by detecting text
      columns on the new image-tabular datasets.
- [ ] **Regenerate `datasets_summary.csv`** and the appendix dataset table for 80 datasets.

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
- [ ] **Re-run every analysis on the 80-dataset collection** wherever the claim is about Full
      rather than Core.
- [ ] **Elo — already complete** (`elo_frozen_vs_tar.csv`, 27 competitors, RandomForest Frozen
      anchored at 1000). Fix the stale "23 competitors" docstring and merge.

## Track 3 — Release

- [x] **Publish the Full-tier text datasets to Kaggle** and load them through the unified API
      (#20). The image half is still pending.
- [ ] Publish the Full-tier image datasets and update this README's dataset counts to 80.
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

Nothing here has been written yet — the paper is untouched. Everything below is a paper-side
consequence of work that has landed in this repo, or a commitment from the rebuttal.

Main-text body is currently over the 10-page camera-ready limit, so every addition needs a
matching trim. Switch `neurips_2026.tex` from `[preprint]` to `[eandd, final]` and compile early
to get a real page count.

## From the Full text half

- [ ] **Report the text half at 40** and describe how the 20 extras were admitted: Joint Signal
      alone (δ=0.001, quorum 3 of 5), measured on the *uploaded* artifacts rather than on the
      pool's original sources.
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

- [ ] Introduce **MulTaBench-Core** (40) and **MulTaBench-Full** (80) in the abstract and §4.
- [ ] **Update every dataset count** across `neurips_2026.tex`, `appendix.tex` (including the
      per-dataset description subsections and the dataset/results tables), and the verbatim
      abstract quote in `checklist.tex`.
- [ ] **Adopt the relaxed trimodal rule** in §4 and Appendix E: report **8 trimodal datasets**,
      keeping the strict-rule result (PetFinder, Amazon Packages) as a stricter sub-tier. Verify
      all 8 pass Joint Signal on both modalities before claiming it. Note that Full is expected
      to reach ~15.
- [ ] Answer veTL's framing question explicitly: we do **not** treat MMTL as two separate bimodal
      problems.

## New appendix material

- [ ] Elo / Bradley–Terry leaderboard table (27 competitors) plus a method description.
- [ ] Per-dataset paired t-test with BH-FDR, naming the 3 non-significant datasets.
- [ ] Pairwise model agreement matrix.
- [ ] Committee simulation detail and the ρ / δ sweep tables.
- [ ] MulTaBench-Full dataset table and its curation record, including the image-tabular
      rejected pool.

## Small fixes found while mapping the paper

- [ ] The trimodal cross-reference in §3 points at `par:text_tabular_curation`; it should be
      `par:image_tabular_curation`.
- [ ] `\subsection{Computation Costs}` carries a `tab:costs` label; rename to `app:costs` (it is
      never referenced, and it collides conceptually with `tab:compute_costs`).
- [ ] `checklist.tex` hardcodes "Section 7" for limitations; this goes stale once a section is added.
- [ ] The appendix dataset table counts 9 image datasets with text columns while the prose says 8
      (`FULLY_MULTIMODAL_DATASET_CANDIDATES` has 8, using a stricter text-column detector).
      Reconcile the two definitions.
- [ ] `paper_production.py` regenerates tables whose captions have since drifted from the
      hand-edited `.tex`, so regenerating will clobber caption edits. Note also that
      `_get_datasets_table_latex()` reads the dataset table *back out of* `appendix.tex`, so that
      one table flows paper → script.
