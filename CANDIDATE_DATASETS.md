# Candidate datasets

Datasets suggested by the community for a future release of MulTaBench. Being listed here doesn't mean a dataset will
be included: each still goes through curation and evaluation first.

To suggest one, open a pull request that adds a row. A good candidate:

- has a supervised target (classification or regression) and at least one image or free-text column that carries
  signal for it, beside the usual numeric and categorical columns;
- comes with a licence that allows redistributing it, since every MulTaBench dataset is hosted on
  [Hugging Face](https://huggingface.co/multabench); say so in the notes if access is restricted;
- isn't already one of the 80 datasets (`MULTABENCH_SOURCES` in `multabench/datasets/all_datasets.py`).

| Dataset | Modalities | Task | Resources | Notes |
|---|---|---|---|---|
| Swiss Real Estate Dataset (SRED) | Tabular, text, property images, satellite images | Regression | [Dataset](https://github.com/Unco3892/SRED_2022), [paper](https://doi.org/10.3390/bdcc6030096) | Swiss real-estate rental prediction from structured property information, descriptions, property images, and satellite images. |
| PetFinder Pawpularity | Tabular, image | Regression | [Dataset](https://www.kaggle.com/competitions/petfinder-pawpularity-score) | Pawpularity score prediction from pet images and associated image metadata. |
| IMDB-WIKI | Tabular, image | Regression | [Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), [paper](https://ieeexplore.ieee.org/document/7406390) | Biological age prediction from face images and associated metadata. |
| MIMIC-IV + MIMIC-CXR | Structured EHR / clinical time series, image, text | Clinical prediction | [MIMIC-IV](https://physionet.org/content/mimiciv/), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) | MIMIC-IV links to MIMIC-CXR, giving structured clinical data with chest X-rays and free-text radiology reports; useful for future clinical multimodal or temporal settings. Requires credentialed PhysioNet access. |
