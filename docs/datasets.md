# Datasets

Every dataset is a public dataset repo in the [`multabench`](https://huggingface.co/multabench)
Hugging Face organization, downloaded on demand. Repos are named
`<core|extended>-<img|text>-<reg|cls>-<name>`; `core` and `extended` are two slices of the
benchmark. Each holds `data.parquet` with explicit column types (datetime,
categorical, numeric, string), `metadata.json` with the target, and, for image datasets, the images
in `images-*.zip`.

The paper used the curated CSV copies on the [`chico89`](https://www.kaggle.com/chico89/datasets)
Kaggle account, which the Hugging Face datasets were converted from. The two may differ: the
Hugging Face datasets store dates, categories and numbers as typed columns where the CSVs hold
strings, and a few non-numeric values in numeric columns were mapped (DVM car's `"1 mile"` to 1,
Anime Planet's `"Unknown"` to missing). Scores on the two versions are not directly comparable.

| What you are looking for | Where it is |
|--------------------------|-------------|
| A dataset's Hugging Face repo | the `MulTaBenchDatasetID` values in `multabench/datasets/all_datasets.py` |
| Where a dataset originally comes from | `MULTABENCH_SOURCES` in `multabench/datasets/all_datasets.py` |
| How a dataset was curated | the recipes in `multabench/benchmark/datasets/` at the [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) tag |
| Size, task and feature counts per dataset | `multabench/leaderboard/results/datasets_summary{,_extra}.csv` at the [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version) tag |

The 40 datasets released alongside the benchmark were admitted on a weaker criterion and carry no
tier name of their own, so their scores must not be pooled with the 40 MulTaBench datasets.

To suggest a dataset for a future release, add it to [`CANDIDATE_DATASETS.md`](../CANDIDATE_DATASETS.md) in a pull
request.
