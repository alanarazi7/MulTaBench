import os
from os.path import exists, join

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.file_downloading import download_url_image_column


DATASET_ID = "REG_IMAGE_AIRBNB_NYC"
SLUG_BASE = "multabench-full-airbnb-nyc"
KAGGLE_SOURCE = "dominoweir/inside-airbnb-nyc"

TARGET_COL = "price"
IMAGE_COL = "listing photo"
_CSV = "listings 2.csv"
_DOWNLOAD_DIR = join(".tabstar_temp_files", "airbnb_nyc_images")
_SAMPLE = 20000
# Airbnb serves multi-megabyte originals and ignores its own size parameters, so 20,000 listings
# would be 140GB of artifact for an encoder that sees 224px. Downscaled on arrival instead.
_MAX_SIDE = 512

# name, description and neighborhood_overview are the host's own copy, and a listing headline
# advertises the very things the photograph shows. The host columns describe the person, the
# scrape columns date the crawl, and the remaining urls identify the row.
_PROSE = ["name", "description", "neighborhood_overview", "host_about", "amenities",
          "bathrooms_text", "host_verifications"]
_IDS = ["id", "listing_url", "scrape_id", "last_scraped", "picture_url", "host_id", "host_url",
        "host_name", "host_thumbnail_url", "host_picture_url", "calendar_last_scraped",
        "license", "calendar_updated"]
# Latitude and longitude pin the address more precisely than a neighbourhood ever could, which
# would let the model price the block rather than the listing.
_GEO = ["latitude", "longitude", "neighbourhood"]
COLS_TO_DROP = _PROSE + _IDS + _GEO


def _shrink(folder: str) -> None:
    """Cap the long side at _MAX_SIDE, in place. Idempotent, so a resumed download is cheap."""
    for name in os.listdir(folder):
        path = join(folder, name)
        try:
            with Image.open(path) as im:
                if max(im.size) <= _MAX_SIDE:
                    continue
                im = im.convert("RGB")
                im.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
                im.save(path, "JPEG", quality=88)
        except Exception:
            continue


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, _CSV), low_memory=False)
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df = df[df[TARGET_COL] > 0]
    df = df[df["picture_url"].astype(str).str.startswith("http")]
    df[IMAGE_COL] = df["picture_url"]
    # A host listing several identical units reuses one photograph.
    df = df.drop_duplicates(subset=IMAGE_COL, keep="first")
    df = df.sample(n=min(_SAMPLE, len(df)), random_state=42).reset_index(drop=True)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df.drop(columns=[c for c in df.columns if df[c].nunique(dropna=True) <= 1],
                 errors="ignore")
    df = download_url_image_column(df=df, img_folder=_DOWNLOAD_DIR, img_col=IMAGE_COL)
    _shrink(_DOWNLOAD_DIR)
    df = df[df[IMAGE_COL].apply(lambda p: bool(p) and exists(join(_DOWNLOAD_DIR, str(p))))]
    df = df.reset_index(drop=True)
    # Nightly rates run to $12,900 with a skew of 17.4; log1p brings that to 0.42.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    assert df[IMAGE_COL].is_unique, "One photo per listing keeps the image split meaningful"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=_DOWNLOAD_DIR,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
