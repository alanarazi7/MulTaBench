# Cached embeddings

The frozen text and image embeddings are cached on Hugging Face, one public dataset repo per
encoder: `multabench/embeddings-e5-small`, `multabench/embeddings-dino-small`, and their large twins
`multabench/embeddings-e5-large` and `multabench/embeddings-dino-large`. A frozen encoder
doesn't depend on the split, so each distinct input of a dataset is embedded once, under
`<dataset>/`, in `embeddings.npz`, which maps every input, exactly as the encoder got it (e5's
`"passage: <column>: <value>"` after a run's missing-text fill, or the image file name), to its raw
encoder output (PCA is fitted per run, on the training rows), and `meta.json`, with the dataset's
rows and columns, the encoding time per feature, the hardware it ran on, the hardware it should run
on (`EMBEDDING_HARDWARE` in `multabench/utils/hardware.py`) and how they differ. A run looks its own
inputs up, so an embedding is never paired with another row's text, and an input missing from the
cache, e.g. after the dataset changed, raises. The cache is optional: `benchmark.py` reads it when
the dataset has embeddings for the encoder, and otherwise embeds live, so new encoders and datasets
need no cache to run.

The fine-tuned encoders, `e5-small-tar` and `dino-small-tar`, are cached in `multabench/embeddings-e5-small-tar` and
`multabench/embeddings-dino-small-tar`. The encoder is fine-tuned with LoRA on a fold's training rows, so each fold
has its own embeddings, under `<dataset>/<size>/fold_<k>/`, with the LoRA adapter beside them and its training
arguments and validation loss in `meta.json`. `e5-small-tar` is tuned on the text datasets and `dino-small-tar` on
the image datasets, on all the dataset's columns of that kind together, so a run with other columns raises. Runs
read these embeddings rather than fine-tuning live, so a tuned encoder needs the cache. The encoder embeds the
training rows it was tuned on, so their embeddings fit the target better than the test rows' do, and the model
fitted on them sees a cleaner signal than it gets at test time. This is kept as is, so all runs share it.

`load_embeddings` returns the embeddings aligned with `load_split`'s rows:

```python
from multabench import Size, load_embeddings

embeddings = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small")
embeddings.train["description"], embeddings.test["description"], embeddings.meta
```

`python -m multabench.embeddings.embed` embeds one dataset, and `python -m
multabench.embeddings.embed_sweep` runs it over encoders × datasets, with the same `--launcher` and
`--sbatch` options as `sweep.py`, and uploads the result with `--upload`:

```bash
python -m multabench.embeddings.embed_sweep --launcher slurm --sbatch="--partition=a100 --gres=gpu:1 --cpus-per-task=6 --mem=32G" --submit
python -m multabench.embeddings.embed_sweep --upload
```

## Pinned revisions

Each package version reads every dataset and embeddings repo at the commit pinned in
`multabench/datasets/revisions.json`, and ships the leaderboard's runs as of its release. After uploading to one of
those repos, `python -m multabench.datasets.revisions` pins them all to their current `main`.
