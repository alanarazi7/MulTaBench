<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/alanarazi7/MulTaBench/master/assets/logo.svg" alt="MulTaBench: Benchmarking Multimodal Tabular Learning" width="640">
  </picture>
</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/multabench/leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86%20Leaderboard-live-06214F" alt="Leaderboard"></a>
  <a href="https://arxiv.org/abs/2605.10616"><img src="https://img.shields.io/badge/arXiv-2605.10616-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2605.10616"><img src="https://img.shields.io/badge/NeurIPS%202026-Spotlight-3D6FC4" alt="NeurIPS 2026 Spotlight"></a>
  <a href="https://huggingface.co/multabench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-80-8EB3EA" alt="Datasets"></a>
  <a href="https://pypi.org/project/multabench/"><img src="https://img.shields.io/pypi/v/multabench?color=06214F" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3D6FC4" alt="Python 3.11+">
  <a href="https://github.com/alanarazi7/MulTaBench/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-8EB3EA" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/multabench/leaderboard"><b>🏆 Leaderboard</b></a> ·
  <a href="https://arxiv.org/abs/2605.10616"><b>📄 Paper</b></a> ·
  <a href="https://huggingface.co/multabench"><b>🤗 Datasets</b></a> ·
  <a href="https://huggingface.co/collections/multabench/multabench-6ad47e4de18e12fe3931a551"><b>📚 Collection</b></a> ·
  <a href="https://github.com/alanarazi7/MulTaBench#getting-started"><b>🚀 Getting started</b></a> ·
  <a href="https://github.com/alanarazi7/MulTaBench#adding-a-model"><b>➕ Add a model</b></a> ·
  <a href="https://github.com/alanarazi7/MulTaBench#citation"><b>📝 Cite</b></a>
</p>

MulTaBench is a benchmark for multimodal tabular learning: tables whose columns hold free text or images alongside
numbers and categories. Its 40 core datasets, 20 text-tabular and 20 image-tabular, were kept only if they show both
**joint signal**, each modality carrying signal about the target that the others don't, and **target awareness**, the
best representation of the text or images depending on the task. 40 more datasets with joint signal are released
alongside them, 80 in total.

Models are ranked on the [live leaderboard](https://huggingface.co/spaces/multabench/leaderboard), but the benchmark's
purpose is to compare ways of encoding text and images for tabular learners, and new methods are welcome.

## Getting started

The core package loads the datasets, their splits and cached embeddings, without torch or any model package:

```bash
pip install multabench
```

```python
from multabench import Size, list_datasets, load_embeddings, load_split, score

datasets = list_datasets(modality="text", task="regression", tier="core")

split = load_split("MUL_IMAGE_PETFINDER", fold=0, size=Size.TEN_K)
split.x_train, split.y_train, split.x_test, split.y_test

embeddings = load_embeddings("MUL_TEXT_WINE_REVIEW", fold=0, size=Size.TEN_K, encoder="e5-small")
embeddings.train["description"], embeddings.test["description"]

predictions = model.fit(split.x_train, split.y_train).predict(split.x_test)
result = score(split.task_type, y_true=split.y_test, y_pred=predictions)
result.test_error
```

`load_published_results()` returns every leaderboard run.

The loaders download from the Hugging Face Hub at pinned revisions. To read a local copy instead, e.g. offline or from
a read-only mount, pass `data_dir`, holding one folder per Hub repo named without the org, with images extracted:
`<data_dir>/<dataset repo>/data.parquet` and `<data_dir>/embeddings-e5-small/<dataset repo>/embeddings.npz`.

```python
split = load_split("MUL_IMAGE_PETFINDER", fold=0, data_dir="/data/multabench")
```

`images=False` loads an image dataset's table without its image files, enough for the cached image embeddings, which
are looked up by file name; the split's `image_folder` is then `None`.

To run the benchmark itself, set up the environment and evaluate one model on one fold:

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env  # Hugging Face token

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0
```

[Running the benchmark][running] covers sweeps, hardware and collecting results; [Cached embeddings][embeddings],
[Datasets][datasets] and [Protocol][protocol], including how Elo is computed and how MulTaBench differs from TabArena,
cover the rest.

## Adding a model

Open a pull request that adds the model; once it is merged, we run it from `master` on the leaderboard's hardware, so
every result comes from the same machines and from code anyone can check out. The pull request adds:

1. A wrapper, `multabench/baselines/<model>.py`: a `TabularModel` subclass, e.g. `multabench/baselines/tabdpt13.py`.
2. The model in `BASELINES` in `benchmark.py`.
3. Its dependencies as an extra in `pyproject.toml`, locked in `requirements.txt`, or in its own
   `requirements-<ENV>.txt` if they conflict.
4. A row in `space/data/models.csv`, with a small logo in `space/assets/logos/` and a link to its paper.

Use the model's default hyperparameters, pin anything it downloads, and say on which datasets it can't run, e.g. too
many classes. [Running the benchmark][adding] has the details.

## Citation

```bibtex
@article{arazi2026multabench,
  title   = {MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image},
  author  = {Arazi, Alan and Shapira, Eilam and Grunblat, Shoham and Ventura, Mor and Hoffer, Elad and Blayer, Gioia and
             Holzm{\"u}ller, David and Purucker, Lennart and Varoquaux, Ga{\"e}l and Hutter, Frank and Reichart, Roi},
  journal = {arXiv preprint arXiv:2605.10616},
  year    = {2026}
}
```

## License

The code is under the [MIT license](https://github.com/alanarazi7/MulTaBench/blob/master/LICENSE); each dataset keeps
the terms of its source, linked in `MULTABENCH_SOURCES` in `multabench/datasets/all_datasets.py`.

[running]: https://github.com/alanarazi7/MulTaBench/blob/master/docs/running.md
[adding]: https://github.com/alanarazi7/MulTaBench/blob/master/docs/running.md#adding-a-model
[embeddings]: https://github.com/alanarazi7/MulTaBench/blob/master/docs/embeddings.md
[datasets]: https://github.com/alanarazi7/MulTaBench/blob/master/docs/datasets.md
[protocol]: https://github.com/alanarazi7/MulTaBench/blob/master/docs/protocol.md
