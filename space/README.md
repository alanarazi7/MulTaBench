---
title: MulTaBench
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: true
short_description: Elo leaderboard for tabular models on text and image data
---

# MulTaBench leaderboard

The leaderboard of [MulTaBench](https://github.com/alanarazi7/MulTaBench): a static page that reads the CSVs in
`data/`, written by `python -m multabench.leaderboard.export` from `multabench/leaderboard/models/`, `models_tar/`
and `models_large/`: each model that reads embeddings is an entry with Frozen, one with TAR and one with Frozen Large
encoders.

To update the Space after the results change:

```bash
python -m multabench.leaderboard.export
huggingface-cli upload multabench/leaderboard space . --repo-type space
```
