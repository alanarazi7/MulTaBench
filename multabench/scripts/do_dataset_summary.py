"""
Verify N and Feat. counts for the MulTaBench-Core (40) or MulTaBench-Full datasets.

Loads each dataset from local Kaggle cache and prints a table that can be
compared against the commented-out tables in neurips_2026.tex.

Usage:
    python do_dataset_summary.py
    python do_dataset_summary.py --tier full
    python do_dataset_summary.py --dry_run          # resolve the dataset list without loading
"""
from __future__ import annotations

import argparse
from os.path import abspath, dirname, join

import pandas as pd

from multabench.datasets.downloading import download_dataset
from multabench.datasets.objects import SupervisedTask
from multabench.datasets.tiers import Tier, datasets_for_tier, tier_from_name
from multabench.baselines.preprocessing.feature_types import detect_image_features
from multabench.preprocessing.feat_types import detect_text_features

_RESULTS_DIR = join(dirname(abspath(__file__)), "..", "leaderboard", "results")
_SUMMARY_CSV = join(_RESULTS_DIR, "datasets_summary.csv")
_FULL_SUMMARY_CSV = join(_RESULTS_DIR, "datasets_summary_full.csv")


def _summary_csv_for_tier(tier: Tier) -> str:
    return _SUMMARY_CSV if tier == Tier.CORE else _FULL_SUMMARY_CSV

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


def summarize_all(tier: Tier = Tier.CORE) -> pd.DataFrame:
    """Load every dataset of the tier and return a summary DataFrame."""
    image_set = set(datasets_for_tier(tier, modality="image"))
    # Text first, then image: this sets the summary CSV's row order.
    all_datasets = datasets_for_tier(tier, modality="text") + datasets_for_tier(tier, modality="image")
    rows = []
    for ds_id in all_datasets:
        name = _FRIENDLY_NAMES.get(ds_id.name, ds_id.name)
        modality = "Image" if ds_id in image_set else "Text"
        try:
            ds = download_dataset(dataset_id=ds_id)
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
    parser = argparse.ArgumentParser(description="Summarize the datasets of a MulTaBench tier")
    parser.add_argument("--tier", type=str, default=Tier.CORE.value, choices=[t.value for t in Tier])
    parser.add_argument("--dry_run", action="store_true", help="print the resolved dataset list and exit")
    args = parser.parse_args()
    tier = tier_from_name(args.tier)

    if args.dry_run:
        # Same order as summarize_all(), so the dry run previews the CSV's row order.
        datasets = datasets_for_tier(tier, modality="text") + datasets_for_tier(tier, modality="image")
        print(f"MulTaBench-{tier.value} — {len(datasets)} datasets")
        for ds_id in datasets:
            print(f"  {ds_id.name}")
        return

    summary_csv = _summary_csv_for_tier(tier)
    print(f"\nGenerating {summary_csv} ...")
    df = summarize_all(tier=tier)
    df.to_csv(summary_csv, index=False)
    print(f"Saved {len(df)} rows → {summary_csv}")


if __name__ == "__main__":
    main()
