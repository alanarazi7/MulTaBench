import os
from os.path import exists, join

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "MUL_IMAGE_HAM10000_LESION"
SLUG_BASE = "multabench-full-ham10000-lesion"
KAGGLE_SOURCE = "kmader/skin-cancer-mnist-ham10000"

TARGET_COL = "dx"
IMAGE_COL = "Lesion Image"
_IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")
# dx_type records how the diagnosis was confirmed, which happens after it is known: a lesion is
# biopsied because it already looks malignant, so histo leaks the target.
COLS_TO_DROP = ["lesion_id", "image_id", "dx_type"]


def _find_image(dir_path: str, image_id: str) -> str | None:
    for sub in _IMAGE_DIRS:
        rel = join(sub, f"{image_id}.jpg")
        if exists(join(dir_path, rel)):
            return rel
    return None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "HAM10000_metadata.csv"))
    # The source photographs a quarter of its lesions more than once. A random split then puts the
    # same lesion on both sides and the image model recognises it rather than the pathology, which
    # is what drove the perfect image-only score this dataset was once excluded on.
    df = df.sort_values("image_id").drop_duplicates("lesion_id", keep="first")
    df[IMAGE_COL] = df["image_id"].apply(lambda i: _find_image(dir_path, i))
    df = df[df[IMAGE_COL].notna()]
    df = df.drop(columns=COLS_TO_DROP, errors="ignore").reset_index(drop=True)
    assert df[IMAGE_COL].is_unique, "One image per lesion is the point of the deduplication"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=dir_path,
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
