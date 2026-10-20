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
`data/`, written by `python -m multabench.leaderboard.export` from `multabench/leaderboard/models/` and
`models_tar/`: each model that reads embeddings is an entry with Frozen and one with TAR encoders.

To update the Space after the results change:

```bash
python -m multabench.leaderboard.export
huggingface-cli upload multabench/leaderboard space . --repo-type space
```

Each change to the board also adds a row to `data/versions.csv`, which the page shows as its version and history.
