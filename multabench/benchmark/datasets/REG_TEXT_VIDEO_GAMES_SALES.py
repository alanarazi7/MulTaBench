"""
Dataset Name: REG_TEXT_VIDEO_GAMES_SALES
====
Examples: 16598
====
URL: https://www.kaggle.com/datasets/gregorut/videogamesales
====
Target Variable: Global_Sales (float64, 623 distinct): ['0.02', '0.03', '0.04', '0.05', '0.01', '0.06', '0.07', '0.08', '0.09', '0.11']
====
Features:

Name (object, 11493 distinct): ['Need for Speed: Most Wanted', 'LEGO Marvel Super Heroes', 'Ratatouille', 'FIFA 14', 'Madden NFL 07', 'LEGO Jurassic World', 'Angry Birds Star Wars', 'LEGO The Hobbit', 'FIFA Soccer 13', 'LEGO Harry Potter: Years 5-7']
Platform (object, 31 distinct): ['DS', 'PS2', 'PS3', 'Wii', 'X360', 'PSP', 'PS', 'PC', 'XB', 'GBA']
Year (float64, 39 distinct, 1.6% missing): ['2009.0', '2008.0', '2010.0', '2007.0', '2011.0', '2006.0', '2005.0', '2002.0', '2003.0', '2004.0']
Genre (object, 12 distinct): ['Action', 'Sports', 'Misc', 'Role-Playing', 'Shooter', 'Adventure', 'Racing', 'Platform', 'Simulation', 'Fighting']
Publisher (object, 578 distinct, 0.3% missing): ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Take-Two Interactive']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_VIDEO_GAMES_SALES"
SLUG_BASE = "multabench-video-games-sales"
KAGGLE_SOURCE = "gregorut/videogamesales"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_SOCIAL_VIDEO_GAMES_SALES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
