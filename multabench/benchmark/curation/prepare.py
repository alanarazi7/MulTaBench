"""
Prepare a curated multimodal dataset for upload to Kaggle.

Output folder structure:
    <output_dir>/<slug>/
        data.csv                  # curated DataFrame with image paths rewritten to images/<basename>
        images/                   # all images, flat, one file per row
        metadata.json             # dataset info: target, image_col, task_type, counts, etc.
        dataset-metadata.json     # Kaggle API metadata (title, id, license)
"""
import pandas as pd
from os import makedirs
from os.path import join
from pandas import DataFrame, Series

from multabench.datasets.description import get_dataset_description
from multabench.datasets.downloading import download_multimodal_dataset
from multabench.datasets.utils import dataset_from_name
from multabench.baselines.preprocessing.feature_types import detect_image_features

from multabench.benchmark.utils.constants import IMAGES_DIR, DATA_CSV
from multabench.benchmark.utils.curation import copy_images, write_metadata, write_kaggle_metadata, task_type_from_name, generate_kaggle_description

_DATASETS_DIR = "multimodal/benchmark/datasets"


def _get_image_col(x: DataFrame) -> str:
    image_cols = detect_image_features(x)
    assert len(image_cols) == 1, f"Expected exactly 1 image column, got: {image_cols}"
    return image_cols[0]


def _drop_null_targets(x: DataFrame, y: Series) -> tuple[DataFrame, Series]:
    null_mask = y.isna()
    n_dropped = null_mask.sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} rows with null target")
        x, y = x[~null_mask], y[~null_mask]
    return x, y


def _normalize_dtypes(df: DataFrame) -> DataFrame:
    """Convert dtypes that don't round-trip cleanly through CSV.

    - datetime -> ISO string
    - boolean  -> int (0/1)
    - everything else stays as-is
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").where(df[col].notna(), other=None)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
    return df


def prepare_dataset(dataset_name: str, slug: str, output_root: str) -> str:
    """
    Download, curate, and prepare a dataset folder ready for Kaggle upload.

    Args:
        dataset_name: e.g. MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY
        slug: Kaggle dataset slug, e.g. multabench-lol-skins
        output_root: root directory for output folders

    Returns the path to the prepared dataset folder.
    """
    dataset_id = dataset_from_name(dataset_name)

    print(f"Downloading and curating {dataset_name}...")
    dataset = download_multimodal_dataset(dataset_id)
    x, y = dataset.x.copy(), dataset.y.copy()
    assert dataset.image_folder is not None, f"{dataset_name} has no image folder"

    x, y = _drop_null_targets(x, y)

    image_col = _get_image_col(x)
    target_name = y.name

    out_dir = join(output_root, slug)
    makedirs(out_dir, exist_ok=True)

    print(f"Copying images from {dataset.image_folder} -> {out_dir}/{IMAGES_DIR}/")
    # Embed y into x before copy_images so that any dropped rows (truncated images)
    # are applied to both x and y together, keeping them in sync.
    x[target_name] = y.values
    x = copy_images(x, image_col, dataset.image_folder, join(out_dir, IMAGES_DIR))
    y = x.pop(target_name)

    df = _normalize_dtypes(x)
    df[target_name] = y.values
    df.to_csv(join(out_dir, DATA_CSV), index=False)
    print(f"Wrote {len(df)} rows to {DATA_CSV}")

    task_type = task_type_from_name(dataset_name)
    write_metadata(out_dir, slug, target_name, image_col, task_type, df)
    kaggle_source = dataset_id.value if hasattr(dataset_id, 'value') and "/" in str(dataset_id.value) else None
    description = generate_kaggle_description(kaggle_source) if kaggle_source else None
    write_kaggle_metadata(out_dir, slug, dataset_name, task_type, description=description)

    url = f"https://www.kaggle.com/datasets/{kaggle_source}" if kaggle_source else None
    annotation = get_dataset_description(name=dataset_name, x=x, y=y, url=url)
    md_path = join(_DATASETS_DIR, f"{dataset_name}.md")
    with open(md_path, "w") as f:
        f.write(annotation)
    print(f"Wrote annotation to {md_path}")

    print(f"\nDataset prepared at: {out_dir}")
    return out_dir
