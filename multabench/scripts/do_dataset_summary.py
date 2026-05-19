"""
Verify N and Feat. counts for the 40 MulTaBench paper datasets.

Loads each dataset from local Kaggle cache and prints a table that can be
compared against the commented-out tables in neurips_2026.tex.

Usage:
    python do_dataset_summary.py
"""
from __future__ import annotations

from os.path import abspath, dirname, join

import pandas as pd

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.image_benchmarks import PAPER_BENCHMARK
from multabench.datasets.objects import SupervisedTask
from multabench.benchmark.load import load_multabench_dataset
from multabench.baselines.preprocessing.feature_types import detect_image_features
from multabench.preprocessing.feat_types import detect_text_features

_SUMMARY_CSV = join(dirname(abspath(__file__)), "..", "leaderboard", "results", "datasets_summary.csv")

TEXT_PAPER_BENCHMARK = [
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING,
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY,
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING,
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY,
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS,
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT,
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES,
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS,
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW,
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES,
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE,
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY,
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE,
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES,
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT,
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES,
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS,
]

_FRIENDLY_NAMES = {
    "BIN_TEXT_FAKE_JOB_POSTING":       "Fake Job Postings",
    "BIN_TEXT_JIGSAW_TOXICITY":        "Jigsaw Toxicity",
    "BIN_TEXT_KICKSTARTER_FUNDING":    "Kickstarter",
    "MUL_TEXT_DATA_SCIENTIST_SALARY":  "Data Scientist Salary",
    "MUL_TEXT_MICHELIN_RESTAURANTS":   "Michelin Guide",
    "MUL_TEXT_PRODUCT_SENTIMENT":      "Product Sentiment",
    "MUL_TEXT_SPOTIFY_GENRES":         "Spotify Genres",
    "MUL_TEXT_US_ACCIDENTS":           "US Accidents",
    "MUL_TEXT_WINE_REVIEW":            "Wine Review",
    "MUL_TEXT_WOMEN_CLOTHING_REVIEW":  "Women's Clothing",
    "REG_TEXT_BABIES_PRICES":          "Baby Products",
    "REG_TEXT_BOOK_PRICE":             "Book Price",
    "REG_TEXT_BOOK_READABILITY":       "Book Readability",
    "REG_TEXT_MERCARI_MARKETPLACE":    "Mercari Marketplace",
    "REG_TEXT_MONTGOMERY_SALARIES":    "Montgomery Salaries",
    "REG_TEXT_ROTTEN_TOMATOES":        "Rotten Tomatoes",
    "REG_TEXT_SCIMAGOJR_IMPACT":       "SciMagojr Impact",
    "REG_TEXT_VANCOUVER_SALARIES":     "Vancouver Salaries",
    "REG_TEXT_VIDEO_GAMES_SALES":      "Video Games Sales",
    "REG_TEXT_ZOMATO_RESTAURANTS":     "Zomato Restaurants",
    "BIN_IMAGE_CELEB_ATTRACTIVENESS":  "Celeb Attractiveness",
    "BIN_IMAGE_HATEFUL_MEME":          "Hateful Meme",
    "BIN_IMAGE_MAMMOGRAPHY_CMMD":      "Mammography CMMD",
    "MUL_IMAGE_CHEXPERT":              "CheXpert",
    "MUL_IMAGE_CBIS_DDSM":             "CBIS-DDSM",
    "MUL_IMAGE_GLAUCOMA_SMDG":         "Glaucoma SMDG",
    "MUL_IMAGE_CSGO_SKIN_PRICE":       "CS:GO Skins",
    "MUL_IMAGE_FLOWER_BOUQUETS":       "Flower Bouquets",
    "MUL_IMAGE_HUBMAP_HPA":            "HubMAP HPA",
    "MUL_IMAGE_JUSTIN_INSTAGRAM":      "Justin Instagram",
    "MUL_IMAGE_PETFINDER":             "PetFinder",
    "MUL_IMAGE_ZOOSCAN_ZOOPLANKTON":   "Zooscan Plankton",
    "REG_IMAGE_AMAZON_BEST_SELLER":    "Amazon Bestseller",
    "REG_IMAGE_AMAZON_PACKAGES":       "Amazon Packages",
    "REG_IMAGE_HNM_FASHION":           "H&M Fashion",
    "REG_IMAGE_KHAADI_CLOTHES":        "Khaadi Clothes",
    "REG_IMAGE_LETTERBOXD_MOVIES":     "Letterboxd Movies",
    "REG_IMAGE_MANGO_MASS":            "Mango Mass",
    "REG_IMAGE_MKPHOTO_BOTS":          "MkPhoto Bots",
    "REG_IMAGE_PAINTING_PRICE":        "Painting Price",
}


def summarize_all() -> pd.DataFrame:
    """Load all 40 benchmark datasets and return a summary DataFrame."""
    image_set = set(PAPER_BENCHMARK)
    all_datasets = list(TEXT_PAPER_BENCHMARK) + list(PAPER_BENCHMARK)
    rows = []
    for ds_id in all_datasets:
        name = _FRIENDLY_NAMES.get(ds_id.name, ds_id.name)
        modality = "Image" if ds_id in image_set else "Text"
        try:
            ds = load_multabench_dataset(ds_id)
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
            continue

        img_cols  = detect_image_features(ds.x)
        text_cols = detect_text_features(ds.x, exclude_columns=set(img_cols))
        n_img     = len(img_cols)
        n_text    = len(text_cols)
        n_struct  = len(ds.x.columns) - n_img - n_text
        n         = len(ds.x)

        is_reg = ds.task_type == SupervisedTask.REGRESSION
        task    = "REG" if is_reg else "CLS"
        classes = None if is_reg else int(ds.y.nunique())

        rows.append({
            "Dataset":   name,
            "Modality":  modality,
            "Task":      task,
            "Classes":   classes,
            "N":         n,
            "Struct.":   n_struct,
            "Text cols": n_text,
            "Img.":      n_img,
        })
        print(f"  ✓ {name}")

    return pd.DataFrame(rows, columns=["Dataset", "Modality", "Task", "Classes", "N",
                                        "Struct.", "Text cols", "Img."])


def main() -> None:
    print("\nGenerating datasets_summary.csv ...")
    df = summarize_all()
    df.to_csv(_SUMMARY_CSV, index=False)
    print(f"Saved {len(df)} rows → {_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
