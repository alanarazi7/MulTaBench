import os
from os.path import exists, join

import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.file_downloading import download_from_figshare, unzip_url_dataset


DATASET_ID = "MUL_IMAGE_HEARTHSTONE_CLASS"
SLUG_BASE = "multabench-full-hearthstone-class"
KAGGLE_SOURCE = "https://figshare.com/ndownloader/files/38561075"

TARGET_COL = "cardClass"
IMAGE_COL = "Image Path"
IMAGE_SUBFOLDER = "Hearthstone-All-cardClass"
_SPLITS = ("train", "dev", "test")
_ARTICLE_ID, _FILE_ID = 21454413, 38561075
# id is a per-row key; collectible is constant once the non-collectible rows are gone.
COLS_TO_DROP = ["id", "collectible"]


def _download(dir_path: str) -> None:
    zip_path = download_from_figshare(article_id=_ARTICLE_ID, file_id=_FILE_ID, path="HearthStoneCards")
    unzip_url_dataset(src_path=zip_path, dst_path=dir_path)
    for split in _SPLITS:
        split_zip = join(dir_path, IMAGE_SUBFOLDER, f"{split}_images.zip")
        unzip_url_dataset(src_path=split_zip)
        os.remove(split_zip)


def _load_and_process(dir_path: str) -> pd.DataFrame:
    main_dir = join(dir_path, IMAGE_SUBFOLDER)
    # The source ships its own train/dev/test split; the benchmark makes its own folds.
    df = pd.concat([pd.read_csv(join(main_dir, f"{s}.csv")) for s in _SPLITS], ignore_index=True)
    # Tokens, boss cards and Battlegrounds/Mercenaries entries are not cards a player collects,
    # and they carry a NONE_cardClass bucket that exists for nothing else.
    df = df[df["collectible"] == 1.0]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[IMAGE_COL].apply(lambda p: exists(join(main_dir, str(p))))]
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    assert df[IMAGE_COL].is_unique, "One row per card art is what makes the image split meaningful"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = join(".tabstar_temp_files", "hearthstone_extracted")
    if not exists(join(dir_path, IMAGE_SUBFOLDER)):
        _download(dir_path)
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
