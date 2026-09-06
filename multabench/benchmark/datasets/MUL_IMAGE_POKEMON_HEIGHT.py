"""
Dataset Name: MUL_IMAGE_POKEMON_HEIGHT
====
Examples: 1077
====
URL: https://www.kaggle.com/datasets/divyanshusingh369/complete-pokemon-library-32k-images-and-csv
====
Target Variable: height_category (str, 10 distinct): ['Quantile 0-10%', 'Quantile 60-70%', 'Quantile 50-60%', 'Quantile 40-50%', 'Quantile 30-40%', 'Quantile 80-90%', 'Quantile 90-100%', 'Quantile 20-30%', 'Quantile 10-20%', 'Quantile 70-80%']
====
Features:

Pokemon (str, 1077 distinct): ['Abomasnow', 'Mega Abomasnow', 'Abra', 'Absol', 'Mega Absol', 'Accelgor', 'Aerodactyl', 'Mega Aerodactyl', 'Aggron', 'Mega Aggron']
Type (str, 207 distinct): ['Water', 'Normal', 'Grass', 'Psychic', 'Electric', 'Fire', 'Normal, Flying', 'Fighting', 'Bug', 'Fairy']
Species (str, 697 distinct): ['Paradox Pokémon', 'Mouse Pokémon', 'Fox Pokémon', 'Dragon Pokémon', 'Flame Pokémon', 'Fossil Pokémon', 'Mushroom Pokémon', 'Balloon Pokémon', 'Seed Pokémon', 'Rabbit Pokémon']
Weight (float64, 500 distinct): ['1.0', '30.0', '4.0', '120.0', '8.0', '15.0', '6.0', '5.0', '12.0', '28.0']
Abilities (str, 648 distinct): ['1. Levitate', '1. Beast Boost', '1. Protosynthesis', '1. Quark Drive', '1. Shed Skin', '1. Clear Body, Light Metal (hidden ability)', '1. Keen Eye, 2. Tangled Feet, Big Pecks (hidden ability)', '1. Soundproof, 2. Static, Aftermath (hidden ability)', '1. Pickup, 2. Gluttony, Quick Feet (hidden ability)', '1. Illusion']
EV Yield (str, 48 distinct): ['2 Attack', '1 Speed', '1 Attack', '2 Speed', '3 Attack', '2 Defense', '3 Sp. Atk', '1 Defense', '2 Sp. Atk', '1 HP']
Catch Rate (str, 36 distinct): ['45 (5.9% with PokéBall, full HP)', '190 (24.8% with PokéBall, full HP)', '255 (33.3% with PokéBall, full HP)', '75 (9.8% with PokéBall, full HP)', '3 (0.4% with PokéBall, full HP)', '120 (15.7% with PokéBall, full HP)', '60 (7.8% with PokéBall, full HP)', '90 (11.8% with PokéBall, full HP)', '30 (3.9% with PokéBall, full HP)', '200 (26.1% with PokéBall, full HP)']
Base Friendship (str, 9 distinct): ['50 (normal)', '35 (lower than normal)', '0 (lower than normal)', '100 (higher than normal)', '—', '140 (higher than normal)', '90 (higher than normal)', '20 (lower than normal)', '70 (higher than normal)']
Base Exp (float64, 192 distinct, 1.0% missing): ['60.0', '62.0', '175.0', '142.0', '61.0', '172.0', '66.0', '168.0', '173.0', '285.0']
Growth Rate (str, 6 distinct): ['Medium Fast', 'Medium Slow', 'Slow', 'Fast', 'Erratic', 'Fluctuating']
Egg Groups (str, 60 distinct): ['Field', 'Undiscovered', 'Bug', 'Mineral', 'Flying', 'Human-Like', 'Amorphous', 'Grass', 'Dragon, Monster', 'Field, Water 1']
Gender (str, 8 distinct): ['50% male, 50% female', 'Genderless', '87.5% male, 12.5% female', '0% male, 100% female', '25% male, 75% female', '100% male, 0% female', '75% male, 25% female', '12.5% male, 87.5% female']
Egg Cycles (str, 12 distinct): ['20 (4,884–5,140 steps)', '15 (3,599–3,855 steps)', '25 (6,169–6,425 steps)', '120 (30,584–30,840 steps)', '40 (10,024–10,280 steps)', '30 (7,454–7,710 steps)', '35 (8,739–8,995 steps)', '10 (2,314–2,570 steps)', '50 (12,594–12,850 steps)', '80 (20,304–20,560 steps)']
HP Base (int64, 107 distinct): ['70', '60', '50', '65', '80', '40', '75', '90', '55', '45']
HP Min (int64, 107 distinct): ['250', '230', '210', '240', '270', '190', '260', '290', '220', '200']
HP Max (int64, 107 distinct): ['344', '324', '304', '334', '364', '284', '354', '384', '314', '294']
Attack Base (int64, 123 distinct): ['100', '80', '65', '85', '60', '75', '55', '50', '90', '45']
Attack Min (int64, 123 distinct): ['184', '148', '121', '157', '112', '139', '103', '94', '166', '85']
Attack Max (int64, 123 distinct): ['328', '284', '251', '295', '240', '273', '229', '218', '306', '207']
Defense Base (int64, 112 distinct): ['60', '50', '80', '70', '40', '65', '90', '100', '45', '75']
Defense Min (int64, 112 distinct): ['112', '94', '148', '130', '76', '121', '166', '184', '85', '139']
Defense Max (int64, 112 distinct): ['240', '218', '284', '262', '196', '251', '306', '328', '207', '273']
Special Attack Base (int64, 122 distinct): ['40', '60', '50', '65', '55', '45', '70', '80', '35', '95']
Special Attack Min (int64, 122 distinct): ['76', '112', '94', '121', '103', '85', '130', '148', '67', '175']
Special Attack Max (int64, 122 distinct): ['196', '240', '218', '251', '229', '207', '262', '284', '185', '317']
Special Defense Base (int64, 103 distinct): ['80', '50', '60', '65', '70', '55', '75', '40', '45', '90']
Special Defense Min (int64, 103 distinct): ['148', '94', '112', '121', '130', '103', '139', '76', '85', '166']
Special Defense Max (int64, 103 distinct): ['284', '218', '240', '251', '262', '229', '273', '196', '207', '306']
Speed Base (int64, 119 distinct): ['50', '60', '65', '30', '70', '45', '90', '85', '40', '80']
Speed Min (int64, 119 distinct): ['94', '112', '121', '58', '130', '85', '166', '157', '76', '148']
Speed Max (int64, 119 distinct): ['218', '240', '251', '174', '262', '207', '306', '295', '196', '284']
pokemon_image (str, 1077 distinct): ['images/Abomasnow_Abomasnow.png', 'images/Mega Abomasnow_Mega Abomasnow.png', 'images/Abra_Abra.png', 'images/Absol_Absol.png', 'images/Mega Absol_Mega Absol.png', 'images/Accelgor_Accelgor.png', 'images/Aerodactyl_Aerodactyl.png', 'images/Mega Aerodactyl_Mega Aerodactyl.png', 'images/Aggron_Aggron.png', 'images/Mega Aggron_Mega Aggron.png']
"""

import os
from os.path import exists, join
from typing import Optional

import kagglehub
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name
from multabench.preprocessing.discretize import discretize_numerical


DATASET_ID = "MUL_IMAGE_POKEMON_HEIGHT"
SLUG_BASE = "multabench-full-pokemon-height"
KAGGLE_SOURCE = "divyanshusingh369/complete-pokemon-library-32k-images-and-csv"

TARGET_COL = "height_category"
IMAGE_COL = "pokemon_image"
IMAGE_SUBFOLDER = join("Pokemon Images DB", "Pokemon Images DB")
MAX_HEIGHT_M = 10  # A handful of Pokemon are far taller than the rest; cap rather than drop.
# Height is what the target is cut from.
COLS_TO_DROP = ["Height"]


def _parse_height(height: str) -> float:
    # '0.6 m (2′00″)'
    if str(height).count("m") != 1:
        raise ValueError(f"Unexpected height format: {height!r}")
    return float(str(height).split("m")[0].strip())


def _parse_kg(weight: str) -> Optional[float]:
    # '135.5 kg (298.7 lbs)'
    if str(weight) == "—":
        return None
    return float(str(weight).split("kg")[0].strip())


def _parse_int(value: str) -> Optional[int]:
    return None if str(value) == "—" else int(value)


def _image_path(dir_path: str, pokemon: str) -> Optional[str]:
    rel = join(pokemon, f"{pokemon}.png")
    return rel if exists(join(dir_path, IMAGE_SUBFOLDER, rel)) else None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "pokemonDB_dataset.csv"))
    assert df["Pokemon"].is_unique, "Expected one row per Pokemon"
    df[IMAGE_COL] = df["Pokemon"].apply(lambda p: _image_path(dir_path, p))
    df = df[df[IMAGE_COL].notna()]
    df["Base Exp"] = df["Base Exp"].apply(_parse_int)
    df["Weight"] = df["Weight"].apply(_parse_kg)
    df["Height"] = df["Height"].apply(_parse_height).clip(upper=MAX_HEIGHT_M)
    df[TARGET_COL] = discretize_numerical(df["Height"])
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
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
