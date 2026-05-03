import os
from os.path import join

import pandas as pd
from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.utils.file_downloading import download_from_figshare, unzip_url_dataset
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY
====
Examples: 1251
====
URL: https://figshare.com/ndownloader/files/38077608
====
Description:
League of Legends (LoL) is a multiplayer online
battle arena (MOBA) video game developed by
Riot Games, Inc. where teams of players compete
in fast-paced matches, utilizing unique champions with distinct abilities to achieve victory. LoL
data is collected from https://lolskinshop.
com/product-category/lol-skins/.

[From paper: https://arxiv.org/pdf/2302.02978]
====
Target Variable: Category (object, 7 distinct): ['Epic Skin ', 'Regular Skin ', 'Legacy Skin ', 'Legendary Skin ', 'Mythic Skin', 'Limited Skin ', 'Others']
====
Features:

SkinName (object, 1251 distinct): ['Boneclaw Shyvana', 'Elderwood Bard', 'Steel Legion Lux', 'Urf Kench', 'Captain Gangplank', 'Birdio', 'Barbecue Leona', 'Iron Solari Leona', 'Bandit Sivir', 'Executioner Mundo']
Price (object, 41 distinct): ['1350', '975', '520', '750', '1820', '10', '3250', '2000', '290', '1850']
Concept (object, 1056 distinct, 0.5% missing): [' High Noon', ' Star Guardian Season 4', ' Ruined', ' Cosmic', ' Arcana', ' Sentinel', ' Space Groove', ' Spirit Blossom', ' Anima Squad', ' Ocean Song']
Model (object, 646 distinct): ['New model, textures and animations!', 'New model and textures!', 'New model and texture!', 'New model', 'New models and textures!', 'All-new model and textures!', 'Outfit changes', 'Completely new model', 'All-new model.', 'New model for Lee Sin.']
Particles (object, 457 distinct): ['No new particles.', 'All new skill particles and animations!', 'AI new skill particles and animations!', 'No new particles', 'New particles for his abilities and auto-attack.', 'New particles for his abilities.', 'New particles for abilities!', 'New VFX', 'New particles for her abilities and auto-attack.', 'New particles!']
Animations (object, 359 distinct): ['No new animations.', 'New recall animation!', 'New animation particles for every skill!', 'New recall animation.', 'No new animations', 'New animations', 'New recall animation', 'No new¬†animations!', 'New animations added', 'Recall animation from their original Star Guardian look!']
Sounds (object, 434 distinct, 0.1% missing): ['No new sounds.', 'New sounds!', 'No new sounds', 'New Sounds', 'New SFX!  New VO processing!', 'No new animations.', 'New recall sounds.', 'New SFX!', 'New SFX! ‚Äì New VO processing!', 'No new SFX']
Sold ingame? (object, 31 distinct): ['Yes', 'Yes, when the legacy vault gets re-opened.', 'No', 'Not released yet', 'Yes, through the hextech crafting system.', 'Yes, when the legacy vault is re-opened', 'Yes, Honor Rewards (Honor capsules)', ' Yes, on Hextech Crafting', 'Yes, through the Hextech crafting system.', ' Yes, when the legacy vault is re-opened']
Image Path (object, 1251 distinct): ['train_images/905.jpg', 'train_images/107.jpg', 'train_images/595.jpg', 'train_images/984.jpg', 'train_images/296.jpg', 'train_images/287.jpg', 'train_images/557.jpg', 'train_images/554.jpg', 'train_images/931.jpg', 'train_images/201.jpg']
Release_date (datetime64[ns], 473 distinct, 1.5% missing): ['2021-01-01 00:00:00', '2020-01-01 00:00:00', '2022-01-01 00:00:00', '2017-12-12 00:00:00', '2019-01-01 00:00:00', '2019-11-27 00:00:00', '2010-12-14 00:00:00', '2012-10-25 00:00:00', '2019-04-23 00:00:00', '2010-08-09 00:00:00']
'''

RELEASE_DATE = "Release date"

def load_df(dir_path: str) -> DataFrame:
    if not os.path.exists(dir_path) or not os.listdir(dir_path):
        _download_and_unzip(dir_path=dir_path)
    df = _collect_split_datasets(dir_path=dir_path)
    df[RELEASE_DATE] = pd.to_datetime(df[RELEASE_DATE], errors='coerce', dayfirst=True)
    return df

def _download_and_unzip(dir_path: str):
    zip_path = download_from_figshare(article_id=21454413, file_id=38077608, path="LeagueOfLegends")
    unzip_url_dataset(src_path=zip_path, dst_path=dir_path)
    for split in ['train', 'dev', 'test']:
        split_zip_name = join(dir_path, IMAGE_FOLDER, f"{split}_images.zip")
        unzip_url_dataset(src_path=split_zip_name)
        os.remove(split_zip_name)

def _collect_split_datasets(dir_path: str) -> DataFrame:
    main_dir = join(dir_path, IMAGE_FOLDER)
    dfs = []
    for split in ['train', 'dev', 'test']:
        split_df = load_csv(dir_path=main_dir, filename=f"{split}.csv")
        dfs.append(split_df)
    df = pd.concat(dfs)
    return df

CONTEXT = "League of Legends Skin category classification"
TARGET = CuratedTarget(raw_name="Category", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ['id']
FEATURES = [CuratedFeature(raw_name="Image Path", feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name=RELEASE_DATE, feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Concept", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Model", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Particles", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Animations", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="Sounds", feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name="SkinName", feat_type=FeatureType.TEXT),
            ]
IMAGE_FOLDER = "LeagueOfLegends-Skin-category"
LOADING_FUNC = load_df
