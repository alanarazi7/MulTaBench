"""
Dataset Name: REG_TEXT_FIFA22_WAGES
====
Examples: 19178
====
URL: https://www.openml.org/search?type=data&id=45012
====
Target Variable: Wage in Euros (float64, 133 distinct): ['2000.0', '500.0', '1000.0', '3000.0', '4000.0', '5000.0', '6000.0', '7000.0', '8000.0', '9000.0']
====
Features:

age (uint8, 29 distinct): ['21', '24', '22', '25', '23', '20', '27', '26', '29', '28']
height_cm (uint8, 49 distinct): ['180', '185', '178', '183', '175', '188', '182', '184', '186', '181']
weight_kg (uint8, 58 distinct): ['70', '75', '80', '72', '73', '74', '78', '76', '77', '68']
nationality_name (category, 163 distinct): ['England', 'Germany', 'Spain', 'France', 'Argentina', 'Brazil', 'Japan', 'Netherlands', 'United States', 'Poland']
overall (uint8, 47 distinct): ['65', '67', '64', '66', '63', '68', '62', '69', '70', '60']
potential (uint8, 46 distinct): ['72', '70', '68', '69', '71', '73', '67', '74', '66', '75']
attacking_crossing (uint8, 88 distinct): ['58', '60', '65', '64', '62', '59', '55', '63', '56', '57']
attacking_finishing (uint8, 94 distinct): ['58', '60', '59', '55', '65', '64', '62', '63', '61', '52']
attacking_heading_accuracy (uint8, 89 distinct): ['58', '55', '62', '59', '60', '64', '65', '56', '54', '57']
attacking_short_passing (uint8, 86 distinct): ['64', '65', '66', '62', '63', '67', '60', '68', '61', '58']
attacking_volleys (uint8, 88 distinct): ['55', '59', '48', '49', '45', '42', '52', '53', '54', '41']
skill_dribbling (uint8, 92 distinct): ['65', '64', '63', '62', '66', '68', '60', '67', '61', '70']
skill_curve (uint8, 89 distinct): ['48', '45', '55', '58', '60', '59', '49', '52', '42', '63']
skill_fk_accuracy (uint8, 90 distinct): ['35', '42', '32', '38', '40', '39', '31', '45', '30', '41']
skill_long_passing (uint8, 85 distinct): ['62', '60', '58', '65', '55', '64', '59', '63', '61', '57']
skill_ball_control (uint8, 88 distinct): ['65', '64', '63', '62', '66', '68', '60', '70', '67', '61']
movement_acceleration (uint8, 84 distinct): ['68', '69', '67', '70', '73', '72', '66', '74', '65', '71']
movement_sprint_speed (uint8, 83 distinct): ['68', '69', '67', '65', '70', '66', '73', '72', '71', '75']
movement_agility (uint8, 79 distinct): ['70', '65', '72', '71', '68', '66', '73', '67', '75', '69']
movement_reactions (uint8, 67 distinct): ['60', '62', '58', '65', '64', '63', '67', '59', '66', '55']
movement_balance (uint8, 79 distinct): ['70', '68', '65', '71', '72', '67', '66', '69', '73', '64']
defending_standing_tackle (uint8, 88 distinct): ['65', '64', '63', '62', '66', '67', '68', '70', '60', '61']
defending_sliding_tackle (uint8, 88 distinct): ['62', '64', '65', '60', '63', '61', '58', '13', '59', '14']
goalkeeping_diving (uint8, 71 distinct): ['8', '14', '7', '9', '11', '13', '10', '12', '6', '15']
goalkeeping_handling (uint8, 69 distinct): ['10', '12', '9', '8', '14', '7', '13', '11', '6', '15']
goalkeeping_kicking (uint8, 79 distinct): ['9', '12', '13', '7', '14', '10', '11', '8', '6', '15']
goalkeeping_positioning (uint8, 77 distinct): ['8', '10', '7', '11', '12', '9', '14', '13', '6', '15']
goalkeeping_reflexes (uint8, 70 distinct): ['9', '11', '7', '8', '13', '10', '14', '12', '6', '15']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_FIFA22_WAGES"
SLUG_BASE = "multabench-full-fifa22-wages"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=45012"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_SPORTS_FIFA22_WAGES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_FIFA22_WAGES for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
