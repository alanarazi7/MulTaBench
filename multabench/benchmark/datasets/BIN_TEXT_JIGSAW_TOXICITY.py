"""
Dataset Name: BIN_TEXT_JIGSAW_TOXICITY
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46654
====
Target Variable: Is Toxic (bool, 2 distinct): ['0', '1']
====
Features:

comment_text (object, 99661 distinct, 0.0% missing): ['No.', 'Well said!', 'Well said.', 'Thank you.', '.', 'Sᴛᴀʀᴛ ᴡᴏʀᴋɪɴɢ ғʀᴏᴍ ʜᴏᴍᴇ! Gʀᴇᴀᴛ ᴊᴏʙ ғᴏʀ sᴛᴜᴅᴇɴᴛs, sᴛᴀʏ-ᴀᴛ-ʜᴏᴍᴇ ᴍᴏᴍs ᴏʀ ᴀɴʏᴏɴᴇ ɴᴇᴇᴅɪɴɢ ᴀɴ ᴇxᴛʀᴀ ɪɴᴄᴏᴍᴇ... Yᴏᴜ ᴏɴʟʏ ɴᴇᴇᴅ ᴀ ᴄᴏᴍᴘᴜᴛᴇʀ ᴀɴᴅ ᴀ ʀᴇʟɪᴀʙʟᴇ ɪɴᴛᴇʀɴᴇᴛ ᴄᴏɴɴᴇᴄᴛɪᴏɴ... Mᴀᴋᴇ $90 ʜᴏᴜʀʟʏ ᴀɴᴅ ᴜᴘ ᴛᴏ $12000 ᴀ ᴍᴏɴᴛʜ ʙʏ ғᴏʟʟᴏᴡɪɴɢ ʟɪɴᴋ ᴀᴛ ᴛʜᴇ ʙᴏᴛᴛᴏᴍ ᴀɴᴅ sɪɢɴɪɴɢ ᴜᴘ... Yᴏᴜ ᴄᴀɴ ʜᴀᴠᴇ ʏᴏᴜʀ ғɪʀsᴛ ᴄʜᴇᴄᴋ ʙʏ ᴛʜᴇ ᴇɴᴅ ᴏғ ᴛʜɪs ᴡᴇᴇᴋ... \n\n+++++++++http://www.cashapp24.com/', 'Thank you!', 'scary area', 'I agree.', 'Why?']
asian (float64, 26 distinct, 77.5% missing): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.5', '0.3', '0.4', '0.6', '0.7']
atheist (float64, 17 distinct, 77.5% missing): ['0.0', '1.0', '0.1', '0.75', '0.8', '0.8333', '0.1667', '0.6', '0.25', '0.7']
bisexual (float64, 17 distinct, 77.5% missing): ['0.0', '0.1', '0.2', '0.1667', '0.3', '0.4', '0.5', '0.3333', '1.0', '0.25']
black (float64, 23 distinct, 77.5% missing): ['0.0', '1.0', '0.8', '0.8333', '0.1', '0.6', '0.5', '0.7', '0.1667', '0.2']
buddhist (float64, 16 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.5', '1.0', '0.8333', '0.6', '0.75', '0.8', '0.2']
christian (float64, 27 distinct, 77.5% missing): ['0.0', '1.0', '0.4', '0.6', '0.3', '0.5', '0.8', '0.1667', '0.2', '0.8333']
female (float64, 33 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.1667', '0.8', '0.2', '0.1', '0.7', '0.3', '0.4']
heterosexual (float64, 20 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '1.0', '0.8333', '0.5', '0.6', '0.25', '0.8', '0.75']
hindu (float64, 17 distinct, 77.5% missing): ['0.0', '0.1667', '0.1', '1.0', '0.2', '0.8333', '0.5', '0.8', '0.25', '0.7']
homosexual_gay_or_lesbian (float64, 23 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.8', '0.1', '0.6', '0.7', '0.2', '0.1667', '0.9']
intellectual_or_learning_disability (float64, 11 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.25', '0.6', '0.5', '0.0008']
jewish (float64, 21 distinct, 77.5% missing): ['0.0', '1.0', '0.1', '0.8', '0.8333', '0.1667', '0.7', '0.2', '0.9', '0.5']
latino (float64, 22 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.4', '0.2', '0.25', '0.3', '1.0', '0.5', '0.6']
male (float64, 36 distinct, 77.5% missing): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.8333', '0.5', '0.8', '0.7', '0.6']
muslim (float64, 25 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.8', '0.5', '0.1', '0.6', '0.2', '0.4', '0.1667']
other_disability (float64, 12 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.1429', '0.0016', '0.0008', '0.0006', '0.4']
other_gender (float64, 9 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.0016', '0.0011', '0.0006', '0.6']
other_race_or_ethnicity (float64, 28 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.4', '0.5', '0.3333', '0.6']
other_religion (float64, 20 distinct, 77.5% missing): ['0.0', '0.1', '0.2', '0.1667', '0.25', '0.3', '0.5', '0.4', '0.3333', '0.75']
other_sexual_orientation (float64, 14 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.1429', '0.3333', '0.0197', '0.0064']
physical_disability (float64, 14 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.5', '0.25', '0.0008', '0.1429']
psychiatric_or_mental_illness (float64, 22 distinct, 77.5% missing): ['0.0', '0.1667', '1.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.8333']
transgender (float64, 21 distinct, 77.5% missing): ['0.0', '0.1', '1.0', '0.2', '0.1667', '0.5', '0.3', '0.25', '0.8333', '0.6']
white (float64, 27 distinct, 77.5% missing): ['0.0', '1.0', '0.8', '0.8333', '0.7', '0.6', '0.1', '0.5', '0.1667', '0.9']
created_date (object, 99996 distinct, 0.0% missing): ['2015-09-29 17:37:25.068440', '2015-10-13 17:16:38.081524', '2015-10-06 18:23:45.995878', '2017-06-22 08:23:15.005369', '2017-04-04 19:01:00.146140', '2017-06-16 15:28:55.285999', '2017-07-25 01:15:39.369564', '2017-01-09 01:29:42.238578', '2016-07-17 22:40:24.664175', '2017-08-27 00:11:36.175241']
funny (int64, 32 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
wow (int64, 8 distinct): ['0', '1', '2', '3', '4', '5', '7', '15']
sad (int64, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
likes (int64, 95 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
disagree (int64, 49 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "BIN_TEXT_JIGSAW_TOXICITY"
SLUG_BASE = "multabench-jigsaw-toxicity"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46654"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.BIN_TEXT_SOCIAL_JIGSAW_TOXICITY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
