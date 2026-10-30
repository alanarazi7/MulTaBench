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
`data/`, written by `python -m multabench.leaderboard.export` from `multabench/leaderboard/models/`, `models_tar/`,
`models_large/` and `models_large_tar/`: each model that reads embeddings is an entry with Frozen-S, one with TAR-S,
one with Frozen-L and one with TAR-L encoders.

To update the Space after the results change:

```bash
python -m multabench.leaderboard.export
huggingface-cli upload multabench/leaderboard space . --repo-type space
```
