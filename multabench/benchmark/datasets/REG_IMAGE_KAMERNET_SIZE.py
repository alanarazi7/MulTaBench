import os
from glob import glob
from os.path import basename, dirname, join

import kagglehub
import numpy as np
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_KAMERNET_SIZE"
SLUG_BASE = "multabench-full-kamernet-size"
KAGGLE_SOURCE = "amancapy/all-kamernet-rent-listings-w-photos-netherlands"

TARGET_COL = "size"
IMAGE_COL = "room pic"
_SUBDIR = "listings_data_11-07-2025"

# link is the listing id, time_of_checking and posted_ago date the scrape rather than the room,
# and description states the floor area outright. location is street-level and nearly unique, so
# the city is lifted out of it first.
COLS_TO_DROP = ["link", "time_of_checking", "posted_ago", "description", "location"]


def _first_photo(dir_path: str) -> dict[int, str]:
    """One photo per listing, in filename order so the choice is stable between runs."""
    out: dict[int, str] = {}
    for path in sorted(glob(join(dir_path, _SUBDIR, "listings_images", "*", "*.png"))):
        out.setdefault(int(basename(dirname(path))), join(*path.split(os.sep)[-2:]))
    return out


def _city(location: str) -> str | None:
    """'ment-rotterdam, bergselaan' -> 'rotterdam'. The street is nearly unique per listing."""
    head = str(location).split(",")[0]
    return head.rsplit("-", 1)[-1].strip() or None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, _SUBDIR, "listings_df.csv"))
    photos = _first_photo(dir_path)
    df[IMAGE_COL] = df["link"].map(photos)
    df = df[df[IMAGE_COL].notna()]
    df["city"] = df["location"].map(_city)
    df = df[pd.to_numeric(df[TARGET_COL], errors="coerce").notna()]
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    # Two landlord columns ship as the placeholder "wip".
    df = df.drop(columns=COLS_TO_DROP + [c for c in df if df[c].nunique() <= 1],
                 errors="ignore").reset_index(drop=True)
    # Rooms run 6 to 442 m2 around a median of 18, so a whole house in the test fold would
    # otherwise decide R2.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    assert df[IMAGE_COL].is_unique, "One photo per listing keeps the image split meaningful"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, _SUBDIR, "listings_images"),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
