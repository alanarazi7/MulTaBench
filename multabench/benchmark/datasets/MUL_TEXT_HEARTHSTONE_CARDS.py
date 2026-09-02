"""
Source: https://www.kaggle.com/jeradrose/hearthstone-cards

2,810 rows. Hearthstone collectible cards; predict which class a card belongs to.

Target: playerClass — 10 classes (NEUTRAL, DRUID, WARRIOR, HUNTER, MAGE, ...).
Features: Card name and flavour/rules text, cost, attack, health, rarity, card type.

Produces:
    <output_dir>/
        data.csv          features + target
        metadata.json     dataset info for MulTaBench loading
        dataset-metadata.json   Kaggle API upload metadata
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_HEARTHSTONE_CARDS"
SLUG_BASE = "multabench-full-hearthstone-cards"
KAGGLE_SOURCE = "https://www.kaggle.com/jeradrose/hearthstone-cards"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_HEARTHSTONE_CARDS for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
