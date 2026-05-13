"""Shared helpers for MulTaBench dataset curation scripts."""
import argparse
import json
import os
import shutil
from datetime import datetime
from os.path import exists, join
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError

from multabench.datasets.description import get_dataset_description
from multabench.benchmark.utils.constants import (
    BENCHMARK_NAME, IMAGES_DIR, KAGGLE_METADATA_JSON, METADATA_JSON, MULTABENCH_KAGGLE_OWNER,
)

TASK_REG = "reg"
TASK_CLS = "cls"


def task_type_from_name(dataset_id: str) -> str:
    prefix = dataset_id.split("_")[0]
    if prefix == "REG":
        return TASK_REG
    if prefix in ("BIN", "MUL"):
        return TASK_CLS
    raise ValueError(f"Cannot infer task type from dataset name: {dataset_id}")


def _modality_from_id(dataset_id: str) -> str:
    parts = dataset_id.split("_")
    if "IMAGE" in parts and "TEXT" in parts:
        return "Image & Text"
    if "IMAGE" in parts:
        return "Image"
    if "TEXT" in parts:
        return "Text"
    return "Multimodal"


def is_valid_curation_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
            img.convert("RGB")
        return True
    except (UnidentifiedImageError, Exception):
        return False


def copy_images(df: pd.DataFrame, image_col: str, src_dir: str, dst_dir: str) -> pd.DataFrame:
    """Copy images to a flat dst_dir and rewrite image_col paths to images/<filename>.
    Rows with truncated or unreadable images are dropped."""
    os.makedirs(dst_dir, exist_ok=True)
    new_paths = []
    bad_indices = []
    for idx, img_path in enumerate(df[image_col]):
        flat_name = str(img_path).replace("/", "_").replace(os.sep, "_")
        src = join(src_dir, str(img_path))
        if not is_valid_curation_image(src):
            print(f"  ⚠️  Skipping truncated/invalid image: {flat_name}")
            bad_indices.append(idx)
            new_paths.append(None)
            continue
        dst = join(dst_dir, flat_name)
        if not exists(dst):
            shutil.copy2(src, dst)
        new_paths.append(f"{IMAGES_DIR}/{flat_name}")
    df = df.copy()
    df[image_col] = new_paths
    if bad_indices:
        print(f"  Dropped {len(bad_indices)} rows with invalid images.")
        df = df[df[image_col].notna()].reset_index(drop=True)
    return df


def write_metadata(output_dir: str, slug: str, target_col: str,
                   image_col: str, task_type: str, df: pd.DataFrame) -> None:
    metadata = {
        "slug": slug,
        "target": target_col,
        "image_col": image_col,
        "task_type": task_type,
        "num_rows": len(df),
        "num_features": len(df.columns) - 1,
    }
    if task_type != TASK_REG:
        metadata["num_classes"] = int(df[target_col].nunique())
    with open(join(output_dir, METADATA_JSON), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {METADATA_JSON}: {metadata}")


def generate_kaggle_description(kaggle_source: str) -> str:
    if kaggle_source.startswith("https://"):
        source_url = kaggle_source
    elif kaggle_source.startswith("c/"):
        source_url = f"https://www.kaggle.com/{kaggle_source}"
    else:
        source_url = f"https://www.kaggle.com/datasets/{kaggle_source}"
    lines = [
        f"This dataset was re-uploaded from [source]({source_url}), potentially with some "
        f"pre-processing. All rights belong to the original authors, as well as its license.",
        "",
        f"It is part of **{BENCHMARK_NAME}**, "
        f"a multimodal tabular benchmark combining images, text, and structured features.",
    ]
    return "\n".join(lines)


def write_kaggle_metadata(output_dir: str, slug: str, dataset_id: str, task_type: str,
                          description: Optional[str] = None) -> None:
    base = slug.removeprefix(f"{BENCHMARK_NAME.lower()}-")
    for suffix in ("-img-cls", "-img-reg", "-txt-cls", "-txt-reg"):
        base = base.removesuffix(suffix)
    readable = base.replace("-", " ").title()
    modality = _modality_from_id(dataset_id)
    task_label = "Reg" if task_type == TASK_REG else "Cls"
    title = f"{BENCHMARK_NAME}: {readable} [{modality}, {task_label}]"
    kaggle_metadata = {
        "title": title,
        "id": f"{MULTABENCH_KAGGLE_OWNER}/{slug}",
        "licenses": [{"name": "other"}],
        "isPublic": True,
    }
    if description:
        kaggle_metadata["description"] = description
    with open(join(output_dir, KAGGLE_METADATA_JSON), "w") as f:
        json.dump(kaggle_metadata, f, indent=2)
    print(f"Wrote {KAGGLE_METADATA_JSON}")


def move_target_last(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols + [target_col]]


_DATASETS_DIR = "multimodal/benchmark/datasets"


def save_dataset(df: pd.DataFrame, output_dir: str, target_col: str, dataset_id: str,
                 slug: str, task_type: str,
                 image_col: Optional[str] = None,
                 kaggle_source: Optional[str] = None) -> None:
    """Write data.csv, metadata.json, dataset-metadata.json, and annotation .md."""
    move_target_last(df, target_col).to_csv(join(output_dir, "data.csv"), index=False)
    print(f"Wrote {len(df)} rows to data.csv")
    write_metadata(output_dir, slug, target_col, image_col, task_type, df)
    description = generate_kaggle_description(kaggle_source) if kaggle_source else None
    write_kaggle_metadata(output_dir, slug, dataset_id, task_type,
                          description=description)
    x = df.drop(columns=[target_col])
    y = df[target_col]
    if not kaggle_source:
        url = None
    elif kaggle_source.startswith("https://"):
        url = kaggle_source
    elif "/" in kaggle_source:
        url = f"https://www.kaggle.com/datasets/{kaggle_source}"
    else:
        url = kaggle_source
    annotation = get_dataset_description(name=dataset_id, x=x, y=y, url=url)
    md_path = join(_DATASETS_DIR, f"{dataset_id}.md")
    with open(md_path, "w") as f:
        f.write(annotation)
    print(f"Wrote annotation to {md_path}")
    print(f"\nDone. Dataset prepared at: {output_dir}")
    print(f"Upload with: python do_kaggle_upload.py --dataset_dir {output_dir}")


def parse_curation_args(slug_base: str, description: str) -> argparse.Namespace:
    """Parse --output_dir and --slug args with timestamped defaults; print resolved values."""
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    default_slug = f"{slug_base}-{timestamp}"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output_dir", default=f"kaggle_uploads/{default_slug}",
                        help=f"Output directory (default: kaggle_uploads/{default_slug})")
    parser.add_argument("--slug", default=default_slug,
                        help=f"Kaggle dataset slug (default: {default_slug})")
    args = parser.parse_args()
    print(f"Slug: {args.slug}")
    print(f"Output dir: {args.output_dir}")
    return args
