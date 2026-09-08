import os
from os.path import exists, join

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from multabench.e5.e5_finetune import get_vanilla_e5, encode_texts_with_e5
from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "BIN_IMAGE_PINTEREST_POPULAR"
SLUG_BASE = "multabench-full-pinterest-popular"
KAGGLE_SOURCE = "andreacombette/pinterest-analysis-using-nlp-and-image-analysis"

TARGET_COL = "widely repinned"
_COUNT_COL = "repin_count"
# Repins run 0 to 280 around a median of 3 with a skew of 11.6, so R2 on the count asks the model
# to hit numbers the data does not support: the regression framing left every state near zero R2
# with folds running negative. A median split is the question this target can actually answer.
IMAGE_COL = "pin"
# Title and description are the only non-image columns, so left as text this source would be an
# image-and-text dataset with no tabular side at all. Encoding them into structured components is
# how BIN_IMAGE_HATEFUL_MEME in Core gives itself one.
_TEXT_COLS = ["title", "description"]
N_E5_DIMS = 15
COLS_TO_DROP = ["Unnamed: 0", "id"]


def _load_and_process(dir_path: str) -> pd.DataFrame:
    csv = [join(r, f) for r, _, fs in os.walk(dir_path) for f in fs if f.endswith(".csv")]
    df = pd.read_csv(sorted(csv)[0], low_memory=False)
    # Images sit two levels down, so the path must be relative to dir_path itself.
    images = {os.path.splitext(f)[0]: os.path.relpath(join(r, f), dir_path)
              for r, _, fs in os.walk(dir_path) for f in fs
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
    df[IMAGE_COL] = df["id"].astype(str).map(images)
    df = df[df[IMAGE_COL].notna()]
    df = df[pd.to_numeric(df[_COUNT_COL], errors="coerce").notna()]
    df[_COUNT_COL] = pd.to_numeric(df[_COUNT_COL])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_vanilla_e5(device)
    text = (df[_TEXT_COLS[0]].fillna("") + ". " + df[_TEXT_COLS[1]].fillna("")).tolist()
    embeddings = encode_texts_with_e5(text, col_name="text", model=model, tokenizer=tokenizer,
                                      device=device)
    reduced = PCA(n_components=N_E5_DIMS, random_state=42).fit_transform(embeddings)
    for i in range(N_E5_DIMS):
        df[f"dim_e5_{i + 1}"] = reduced[:, i]

    df[TARGET_COL] = (df[_COUNT_COL] > df[_COUNT_COL].median()).map({True: "yes", False: "no"})
    df = df.drop(columns=_TEXT_COLS + COLS_TO_DROP + [_COUNT_COL],
                 errors="ignore").reset_index(drop=True)
    assert df[IMAGE_COL].is_unique, "One pin per image keeps the image split meaningful"
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
