# `text_source/` — source-bench (dropped-dataset) results

Per-dataset Frozen / TAR leaderboard results for the **36 source-bench text datasets**
(the text datasets that did not enter the curated MulTaBench-20; they do not overlap the
20 in `../text/`). Together `text/` (20) + `text_source/` (36) = the full 56-dataset text
pool.

- **One CSV per dataset**, same schema as `../text/`:
  `Name, test_score, fold, multimodal_state, model, Created, dataset, Runtime, Hostname, State`.
- `multimodal_state`: `all` = **Frozen** (frozen e5-small vectors), `ft` = **TAR**
  (target-aware LoRA-tuned e5). `no_text` / `text_only` ablations are not included here.
- All **12 panel models** are present (TabPFN-v2/v2p5/v3, TabICLv2, TabDPT, TabM, RealMLP,
  XGBoost, CatBoost, LightGBM, RandomForest, TabFM), 5 folds each.
- Coverage is complete except structurally-unrunnable cells: **TabFM** omits 3 datasets that
  OOM even at 98 GB (`MUL_TEXT_HOUSES_MELBOURNE_AIRBNB`,
  `REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO`, `REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020`).
- Encoder: e5-small-v2 (matches `../text/` + `../more_baselines/`).
