import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_WATCH_TIER"
SLUG_BASE = "multabench-full-watch-tier"
KAGGLE_SOURCE = "mathewkouch/a-dataset-of-watches"

TARGET_COL = "tier_premium"
PRICE_COL = "price"
BRAND_COL = "brand"
# Below this a brand's median is its own single watch, making the target trivially 1.0.
MIN_PER_BRAND = 5
# Dividing by a brand's own median erases every trace of brand from the tabular side, which
# turns Joint Signal into an image-only test. Price tiers normalise the lookup away while the
# brand still says where a watch sits inside its tier.
PRICE_TIERS = 5
IMAGE_COL = "watch"
_BASE = join("watches", "watches")
IMAGE_SUBFOLDER = join(_BASE, "images")


def _usd_to_float(usd: str) -> float:
    return float(str(usd).replace("$", "").replace(",", "").strip())


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, _BASE, "metadata.csv"))
    df = df.rename(columns={"image_name": IMAGE_COL}).drop(columns=["Unnamed: 0"], errors="ignore")
    df[PRICE_COL] = df[PRICE_COL].apply(_usd_to_float)
    df = df[df[IMAGE_COL].apply(lambda i: exists(join(dir_path, IMAGE_SUBFOLDER, str(i))))]
    df = df[df[PRICE_COL].notna()]
    # The same model is listed several times at one price, so brand and name alone memorise it.
    df = df.drop_duplicates(subset=[BRAND_COL, "name"], keep="first")
    df = df[df.groupby(BRAND_COL)[BRAND_COL].transform("size") >= MIN_PER_BRAND]
    # Brand alone explains 81% of price, so it answers the question before the photo is seen.
    # Pricing against the tier median asks what makes this watch dear for its price bracket.
    brand_median = df.groupby(BRAND_COL)[PRICE_COL].median()
    tier = pd.qcut(brand_median, PRICE_TIERS, labels=False, duplicates="drop")
    df["price_tier"] = df[BRAND_COL].map(tier)
    df[TARGET_COL] = df[PRICE_COL] / df.groupby("price_tier")[PRICE_COL].transform("median")
    df = df.drop(columns=["price_tier"])
    df = df.drop(columns=[PRICE_COL]).reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
