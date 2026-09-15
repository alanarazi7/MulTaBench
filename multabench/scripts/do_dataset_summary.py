"""
Verify N and Feat. counts for the MulTaBench paper datasets.

Loads each dataset from local Kaggle cache and writes one summary CSV for the
40 benchmark datasets and one for the 40 additional released datasets, both
consumed by the paper-production tab.

Usage:
    python do_dataset_summary.py
"""
from __future__ import annotations

from os.path import abspath, dirname, join

import pandas as pd

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.all_multabench_datasets import MULTABENCH_CORE_IMAGE, MULTABENCH_CORE_TEXT
from multabench.datasets.objects import SupervisedTask
from multabench.benchmark.load import load_multabench_dataset
from multabench.baselines.preprocessing.feature_types import detect_image_features
from multabench.preprocessing.feat_types import detect_text_features

_RESULTS_DIR = join(dirname(abspath(__file__)), "..", "leaderboard", "results")
_SUMMARY_CSV = join(_RESULTS_DIR, "datasets_summary.csv")
_SUMMARY_EXTRA_CSV = join(_RESULTS_DIR, "datasets_summary_extra.csv")


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


_FRIENDLY_NAMES_EXTRA_IMAGE = {
    "BIN_IMAGE_OASIS_ALZHEIMERS":     "OASIS Alzheimer's",
    "BIN_IMAGE_PINTEREST_POPULAR":    "Pinterest Repins",
    "MUL_IMAGE_HAM10000_LESION":      "HAM10000 Lesions",
    "MUL_IMAGE_HEARTHSTONE_CLASS":    "Hearthstone Class",
    "MUL_IMAGE_MINECRAFT":            "Minecraft Mobs",
    "MUL_IMAGE_PAD_UFES_LESION":      "PAD-UFES Lesions",
    "MUL_IMAGE_POKEMON_HEIGHT":       "Pokemon Height",
    "MUL_IMAGE_REDDIT_MEMES":         "Reddit Memes",
    "REG_IMAGE_AIRBNB_NYC":           "Airbnb NYC",
    "REG_IMAGE_DVM_CAR":              "DVM Car Prices",
    "REG_IMAGE_FLIPKART_RATIO":       "Flipkart Discount",
    "REG_IMAGE_GOIAS_HOUSES":         "Goias Apartments",
    "REG_IMAGE_KAMERNET_SIZE":        "Kamernet Room Size",
    "REG_IMAGE_LAHAINA_AUCTION":      "Lahaina Art Auction",
    "REG_IMAGE_ROMANIA_PRICE":        "eMAG Products",
    "REG_IMAGE_SAO_PAULO_HOUSES":     "Sao Paulo Apartments",
    "REG_IMAGE_SOCAL_HOUSES":         "SoCal Houses",
    "REG_IMAGE_TOKOPEDIA_WEIGHT":     "Tokopedia Weight",
    "REG_IMAGE_WATCH_TIER":           "Watch Tier",
    "REG_IMAGE_ZEPTO_PRICE":          "Zepto Groceries",
}

_FRIENDLY_NAMES_EXTRA_TEXT = {
    "BIN_TEXT_CALIFORNIA_PRICES":     "California Prices",
    "BIN_TEXT_IMDB_GENRE":            "IMDB Genre",
    "BIN_TEXT_OSHA_INJURY":           "OSHA Injury",
    "MUL_TEXT_AMERICAN_EAGLE_PRICES": "American Eagle",
    "MUL_TEXT_BOOKS_GOODREADS":       "Goodreads Books",
    "MUL_TEXT_BOX_OFFICE":            "Box Office",
    "MUL_TEXT_CONSUMER_COMPLAINT":    "Consumer Complaints",
    "MUL_TEXT_KOREAN_DRAMA":          "Korean Drama",
    "MUL_TEXT_MELBOURNE_AIRBNB":      "Airbnb Melbourne",
    "MUL_TEXT_NEWS_CHANNEL":          "News Channel",
    "REG_TEXT_AIRBNB_SEATTLE":        "Airbnb Seattle",
    "REG_TEXT_ANIME_PLANET":          "Anime Planet",
    "REG_TEXT_CHOCOLATE_BAR_RATINGS": "Chocolate Bars",
    "REG_TEXT_FIFA22_WAGES":          "FIFA22 Wages",
    "REG_TEXT_RAMEN_RATINGS":         "Ramen Ratings",
    "REG_TEXT_USED_CAR_PAKISTAN":     "Used Cars Pakistan",
    "REG_TEXT_USED_CAR_SAUDI":        "Used Cars Saudi",
    "REG_TEXT_WIKILIQ_PRICES":        "WikiLiq Spirits",
    "REG_TEXT_WINE_POLISH_MARKET":    "Wine Poland",
    "REG_TEXT_WINE_VIVINO_SPAIN":     "Wine Vivino Spain",
}


def _ids(names: dict[str, str]) -> list[MulTaBenchDatasetID]:
    return [MulTaBenchDatasetID[name] for name in names]


def summarize(image_ids, text_ids, friendly: dict[str, str]) -> pd.DataFrame:
    """Load each dataset and return a summary DataFrame, image panel first."""
    image_set = set(image_ids)
    rows = []
    for ds_id in list(text_ids) + list(image_ids):
        name = friendly.get(ds_id.name, ds_id.name)
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
    for path, image_ids, text_ids, friendly in (
        (_SUMMARY_CSV, MULTABENCH_CORE_IMAGE, MULTABENCH_CORE_TEXT, _FRIENDLY_NAMES),
        (_SUMMARY_EXTRA_CSV, _ids(_FRIENDLY_NAMES_EXTRA_IMAGE), _ids(_FRIENDLY_NAMES_EXTRA_TEXT),
         {**_FRIENDLY_NAMES_EXTRA_IMAGE, **_FRIENDLY_NAMES_EXTRA_TEXT}),
    ):
        print(f"\nGenerating {path} ...")
        df = summarize(image_ids=image_ids, text_ids=text_ids, friendly=friendly)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows → {path}")


if __name__ == "__main__":
    main()
