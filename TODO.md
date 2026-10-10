# TODO

Work in progress towards the first release of the living leaderboard. Items are removed from this
file as the PRs that complete them are merged.

## Embedding cache

- [ ] Compute and upload the `dino-small` embeddings for the benchmark datasets with
      `python -m multabench.embeddings.embed_sweep`, then pin `REVISIONS` in
      `multabench/embeddings/hub.py` to the uploaded commit.
- [ ] The 62 uploaded `e5-small` `meta.json` files store `expected_hardware` as `{gpu, cpu, cpus, ram_gb}`,
      while new ones write `{gpu_prefix, cpu_prefix, cpus, ram_gb}`. Rename the keys in the uploaded files
      (a metadata-only upload, then a new pin in `REVISIONS`), or accept the difference.
- [ ] Cache the `e5-small-tar` and `dino-small-tar` embeddings: one fine-tune per (size, fold), each
      seeing only its fold's training rows, uploaded with its LoRA adapter.
- [ ] Make the model runners read the cached embeddings instead of re-embedding, so one fine-tune per
      fold is shared by every model. Each run records how many features had no cached embedding and
      were encoded live.

## Runs

- [ ] Move `benchmark.py`, `sweep.py` and `collect_results.py` into the package, like
      `multabench.embeddings.embed`, so they run with `python -m` without a checkout.
- [ ] Split `benchmark.py` into a few functions in `multabench/benchmark/` (argument parsing,
      encoder resolution, running and writing the result), leaving the script a thin entry point.
- [ ] Run every leaderboard entry on the official hardware.
- [ ] `sweep.py` takes one `--sbatch`, so a sweep mixing CPU and GPU runs has to be split in two by
      hand. Group its runs by `run_hardware` and take a second set of sbatch arguments for CPU runs.
- [ ] End-to-end models, which take the raw table without our preprocessing: TabSTAR, ConTextTab and
      AutoGluon multimodal.
- [ ] Rerun every result CSV under `multabench/leaderboard/results/` with the TabArena metrics; the
      current ones still report AUC and R².

## Leaderboard

- [ ] Elo leaderboard as in TabArena: `bencheval`, RandomForest (default) anchored at 1000, missing
      results imputed with RandomForest and flagged, bootstrapped confidence intervals. Update the
      "Leaderboard" row of the README table.
- [ ] Coverage column (share of (dataset, fold) cells each model completed), plus win rate, average
      rank and improvability as secondary columns.
- [ ] Leaderboard tab reading the new results CSV.
- [ ] README section on submitting a model: open a PR with the model's wrapper (and its
      dependencies); maintainers rerun it on the official hardware, and self-reported results are
      not accepted.
- [ ] Remove the remaining paper analysis code in `multabench/leaderboard/`, keeping only the main
      leaderboard tab.

## Shared fine-tuning code

`e5/e5_finetune.py` and `dino/dino_finetune.py` are near copies. Move what they share into
`finetune/`, keeping only the encoder-specific parts (loading, the dataset class, pooling) per
modality:

- [ ] `_get_lora_target_modules` and `_print_trainable_params_per_layer`, which differ only in the
      module names.
- [ ] `E5ForTuning` / `DINOForTuning`: the same backbone + linear head, apart from pooling.
- [ ] `finetune_*_with_lora`: the same seeding, `LoraConfig`, `TrainingArguments`, early stopping
      and `Trainer` loop.
- [ ] `DinoTrainArgs` / `E5TrainArgs`, and the matching `--dino_*` / `--e5_*` flags and kwargs
      dicts in `benchmark.py`.

## Later, not needed for the first release

- Minimal installation for `load_split` and `load_embeddings`.
- Hosting the leaderboard as a Hugging Face Space.
- `e5-large` and `dino-large` stay available as options but are not run.
- `--size full` results.
- A time limit per (model, dataset, fold), excluding encoder fine-tuning, with a result row with
  `"status": "timeout"`.
- macOS: torch and LightGBM/XGBoost each bundle their own libomp, and the import order decides
  which one breaks.
