"""
Dataset Name: BIN_IMAGE_HATEFUL_MEME
====
Examples: 10000
====
URL: https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset
====
Target Variable: label (float64, 2 distinct, 10.0% missing): ['0.0', '1.0']
====
Features:

img (object, 10000 distinct): ['images/img_42953.png', 'images/img_23058.png', 'images/img_13894.png', 'images/img_37408.png', 'images/img_82403.png', 'images/img_16952.png', 'images/img_76932.png', 'images/img_70914.png', 'images/img_02973.png', 'images/img_58306.png']
dim_e5_1 (float32, 8013 distinct): ['-0.0487', '-0.2188', '0.1389', '-0.0636', '-0.0504', '0.0428', '-0.0721', '-0.1617', '0.069', '-0.2133']
dim_e5_2 (float32, 8015 distinct): ['0.1035', '0.022', '-0.0133', '0.1205', '-0.0513', '0.0002', '-0.0522', '0.0305', '-0.1024', '-0.0132']
dim_e5_3 (float32, 8012 distinct): ['0.0638', '-0.0537', '-0.0914', '-0.0037', '-0.0845', '-0.0458', '0.061', '-0.0199', '0.026', '0.0397']
dim_e5_4 (float32, 8013 distinct): ['-0.114', '0.0997', '0.1609', '0.0295', '0.1247', '-0.1236', '-0.0805', '0.0666', '0.0623', '0.1209']
dim_e5_5 (float32, 8015 distinct): ['-0.1173', '-0.0484', '-0.0062', '-0.056', '0.0208', '0.0018', '-0.0798', '-0.0138', '-0.0147', '-0.0229']
dim_e5_6 (float32, 8014 distinct): ['0.0102', '0.0061', '-0.0299', '-0.0572', '0.0387', '0.0581', '0.034', '0.1068', '0.0097', '-0.0113']
dim_e5_7 (float32, 8015 distinct): ['-0.0743', '-0.0368', '0.0274', '0.105', '0.0022', '-0.0832', '0.0234', '0.0314', '0.1027', '-0.0475']
dim_e5_8 (float32, 8013 distinct): ['-0.0007', '0.0115', '0.0424', '-0.1109', '-0.0544', '-0.0169', '0.0134', '0.004', '0.1289', '0.0753']
dim_e5_9 (float32, 8013 distinct): ['0.0682', '0.0734', '-0.0257', '-0.0297', '-0.0238', '-0.0731', '0.0302', '-0.0765', '0.0739', '-0.0469']
dim_e5_10 (float32, 8014 distinct): ['-0.0771', '0.1078', '-0.0881', '-0.0257', '-0.0247', '0.0024', '-0.0425', '0.0706', '0.0823', '-0.0838']
dim_e5_11 (float32, 8013 distinct): ['0.019', '-0.0486', '0.017', '-0.0139', '0.0122', '0.0046', '-0.0881', '-0.0221', '0.0812', '0.0343']
dim_e5_12 (float32, 8015 distinct): ['0.0957', '0.0', '-0.0446', '0.0344', '-0.0197', '0.005', '-0.0223', '0.0193', '-0.0475', '0.0636']
dim_e5_13 (float32, 8013 distinct): ['-0.0745', '0.0435', '0.0357', '0.0083', '0.0333', '0.0788', '0.0463', '-0.0123', '-0.0006', '0.04']
dim_e5_14 (float32, 8014 distinct): ['0.0068', '0.0091', '0.0402', '-0.0728', '-0.0261', '-0.0248', '0.0179', '-0.0081', '-0.0016', '0.0107']
dim_e5_15 (float32, 8014 distinct): ['-0.0969', '0.032', '0.0174', '0.003', '0.051', '-0.0734', '-0.0809', '-0.0347', '0.1287', '-0.0229']
dim_e5_16 (float32, 8012 distinct): ['-0.0259', '-0.0436', '0.0097', '0.0197', '-0.0198', '-0.0154', '0.0481', '-0.0706', '-0.0188', '-0.0979']
dim_e5_17 (float32, 8013 distinct): ['0.0326', '0.0292', '-0.0503', '0.0334', '0.0119', '0.1159', '-0.1456', '-0.0218', '-0.038', '0.0142']
dim_e5_18 (float32, 8015 distinct): ['-0.0816', '-0.0272', '-0.0182', '-0.0657', '0.0091', '0.0188', '-0.0663', '0.0334', '0.0844', '0.0225']
dim_e5_19 (float32, 8014 distinct): ['-0.0944', '0.0529', '0.0514', '-0.0116', '-0.0288', '0.0034', '-0.0223', '0.0369', '-0.0124', '-0.0271']
dim_e5_20 (float32, 8015 distinct): ['-0.0791', '0.0334', '-0.0061', '0.0396', '-0.0066', '-0.0379', '-0.0539', '0.0348', '0.0056', '-0.0149']
"""

import os
from os.path import join

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from multabench.e5.e5_finetune import get_vanilla_e5, encode_texts_with_e5
from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.utils.io_handlers import load_json_lines


DATASET_ID = "BIN_IMAGE_HATEFUL_MEME"
SLUG_BASE = "multabench-hateful-meme"
KAGGLE_SOURCE = "parthplc/facebook-hateful-meme-dataset"

TARGET_COL = "label"
IMAGE_COL = "img"
DATA_SUBFOLDER = "data"
N_E5_DIMS = 20


def _load_and_process(dir_path: str) -> pd.DataFrame:
    main_path = join(dir_path, DATA_SUBFOLDER)
    rows = []
    for split in ["train", "dev", "test"]:
        split_path = join(main_path, f"{split}.jsonl")
        for d in load_json_lines(split_path):
            d.pop("id", None)
            rows.append(d)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[hateful_meme] loaded {len(df)} rows (dropped unlabeled test rows), encoding text with E5...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_vanilla_e5(device)
    texts = df["text"].fillna("").tolist()
    embeddings = encode_texts_with_e5(texts, col_name="text", model=model, tokenizer=tokenizer, device=device)
    print(f"[hateful_meme] E5 done ({embeddings.shape}), running PCA to {N_E5_DIMS} dims...")
    pca = PCA(n_components=N_E5_DIMS, random_state=42)
    reduced = pca.fit_transform(embeddings)
    for i in range(N_E5_DIMS):
        df[f"dim_e5_{i + 1}"] = reduced[:, i]
    df = df.drop(columns=["text"])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    # img paths in jsonl are like "img/42953.png" relative to the data/ folder
    df = copy_images(df=df, image_col=IMAGE_COL,
                     src_dir=join(dir_path, DATA_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL,
                 dataset_id=DATASET_ID, slug=slug, image_col=IMAGE_COL,
                 task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate Facebook Hateful Meme dataset for MulTaBench")
    curate(output_dir=args.output_dir, slug=args.slug)
