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

20 image benchmark datasets hosted on Kaggle under `multabench-*`. Downloads are automatic via `kagglehub`. Datasets follow the naming convention `{TASK}_{MODALITY}_{NAME}` where task is `BIN`/`MUL`/`REG`.

**Image benchmark** (20 datasets): celebrity attractiveness, hateful memes, mammography, CheXpert, CBIS-DDSM, glaucoma, CS:GO skins, flower bouquets, HuBMAP, Instagram engagement, PetFinder adoption, zooplankton, Amazon bestsellers, Amazon packages, H&M fashion, Khaadi clothes, Letterboxd movies, mango mass, photography bots, painting price.

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
