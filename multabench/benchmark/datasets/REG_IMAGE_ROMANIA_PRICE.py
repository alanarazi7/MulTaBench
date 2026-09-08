import os
from os.path import exists, join

import kagglehub
import numpy as np
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.file_downloading import url_to_filename


DATASET_ID = "REG_IMAGE_ROMANIA_PRICE"
SLUG_BASE = "multabench-full-romania-price"
KAGGLE_SOURCE = "furduisorinoctavian/romanian-products-from-emag-and-flanco"

TARGET_COL = "price"
IMAGE_COL = "product pic"
IMAGE_SUBFOLDER = "downloaded_romania_images"
# The product title names the object, and naming the object is most of the price: leaving it in
# would let the text encoder answer the question the photograph is being asked. It is also the
# only Romanian free text here; category and shop are closed vocabularies.
COLS_TO_DROP = ["id", "name", "url", "image_url"]


def _parse_lei(price: str) -> float | None:
    """'6.999,99 ' -> 6999.99. Romanian writes the thousands separator as a dot."""
    try:
        return float(str(price).strip().replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "products.csv"))
    df[TARGET_COL] = df[TARGET_COL].map(_parse_lei)
    df = df[df[TARGET_COL].notna()]
    df[IMAGE_COL] = df["image_url"].astype(str).map(url_to_filename)
    df = df[df[IMAGE_COL].apply(lambda p: exists(join(dir_path, IMAGE_SUBFOLDER, p)))]
    # Half the catalogue reuses a stock photo across variants of one product; a shared image would
    # otherwise sit on both sides of a random split.
    df = df.drop_duplicates(subset=IMAGE_COL, keep="first")
    df = df[df[TARGET_COL] > 0]
    # Only eMag listings carry a usable photo, which leaves shop constant.
    df = df.drop(columns=COLS_TO_DROP + [c for c in df if df[c].nunique() <= 1],
                 errors="ignore").reset_index(drop=True)
    # Prices span four orders of magnitude, from cables to televisions.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    assert df[IMAGE_COL].is_unique, "One product per photo keeps the image split meaningful"
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
