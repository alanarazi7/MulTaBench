from os.path import exists, join

from pandas import DataFrame, read_csv

from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.kaggle_competitions import download_kaggle_competition, KAGGLE_CACHE_DIR
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.utils.io_handlers import unzip_file


COMPETITION_NAME = "petfinder-adoption-prediction"
COMPETITION_FOLDER = join(KAGGLE_CACHE_DIR, COMPETITION_NAME)

'''
Dataset Name: c/petfinder-adoption-prediction/
====
Examples: 14652
====
URL: https://www.kaggle.com/c/petfinder-adoption-prediction
====
Bag Of Tricks Paper:
petfinder: Predict the speed at which a pet is adopted, based on the its image and metadata including
text description, health situation, fur length, color and so on. This dataset originally stems from
https://www.kaggle.com/competitions/petfinder-adoption-prediction. We randomly split the original
training set at 4:1 ratio for new training and test sets.

Description:
In this competition you will predict the speed at which a pet is adopted, based on the pet's listing
on PetFinder. The data included text, tabular, and image data.

AdoptionSpeed values:
0 - Pet was adopted on the same day as it was listed.
1 - Pet was adopted between 1 and 7 days after being listed.
2 - Pet was adopted between 8 and 30 days after being listed.
3 - Pet was adopted between 31 and 90 days after being listed.
4 - No adoption after 100 days of being listed.
====
Target Variable: AdoptionSpeed (object, 5 distinct): ['8-30 Days', 'Not adopted in 100 days', '31-90 Days', '1-7 Days', 'Same Day']
====
Features:

Type (int64, 2 distinct): ['1', '2']
Name (object, 8911 distinct): ['Unknown Value', 'Baby', 'Lucky', 'Brownie', 'Mimi']
Age (int64, 63 distinct): [...]
Breed1 (object, 175 distinct): ['Mixed Breed', 'Domestic Short Hair', 'Labrador Retriever']
Breed2 (object, 132 distinct): ['', 'Mixed Breed', 'Domestic Short Hair']
Gender (int64, 3 distinct): ['2', '1', '3']
Color1 (object, 7 distinct): ['Black', 'Brown', 'Golden', 'Cream', 'Gray', 'White', 'Yellow']
Color2 (object, 7 distinct): ['', 'White', 'Brown', 'Cream', 'Gray', 'Yellow', 'Golden']
Color3 (object, 6 distinct): ['', 'White', 'Cream', 'Gray', 'Yellow', 'Golden']
MaturitySize (int64, 4 distinct): ['2', '1', '3', '4']
FurLength (int64, 3 distinct): ['1', '2', '3']
Vaccinated (int64, 3 distinct): ['2', '1', '3']
Dewormed (int64, 3 distinct): ['1', '2', '3']
Sterilized (int64, 3 distinct): ['2', '1', '3']
Health (int64, 3 distinct): ['1', '2', '3']
Quantity (int64, 19 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Fee (int64, 74 distinct): ['0', '50', '100', '200', '150', '20', '300', '30', '250', '1']
State (object, 14 distinct): ['Selangor', 'Kuala Lumpur', 'Pulau Pinang', 'Johor', 'Perak']
VideoAmt (int64, 9 distinct): ['0', '1', '2', '3', '4', '5', '6', '8', '7']
Description (object, 13719 distinct): [long pet descriptions ...]
PhotoAmt (float64, 30 distinct): ['1.0', '2.0', '3.0', '5.0', '4.0', '6.0', '7.0']
Pet Image (object, 14652 distinct): ['86e1089a3-1.jpg', '6296e909a-1.jpg', ...]
'''

LABEL_NAME = "AdoptionSpeed"
PET_IMAGE_COLUMN = 'Pet Image'


def load_df(dir_path: str) -> DataFrame:
    if not exists(COMPETITION_FOLDER):
        download_kaggle_competition(competition=COMPETITION_NAME)
        zip_file = f"{COMPETITION_FOLDER}.zip"
        unzip_file(zip_file)
    df = _get_csv("train/train.csv")
    breed = _get_csv("breed_labels.csv")
    breed = breed.set_index('BreedID')['BreedName'].to_dict()
    color = _get_csv("color_labels.csv")
    color = color.set_index('ColorID')['ColorName'].to_dict()
    state = _get_csv("state_labels.csv")
    state = state.set_index('StateID')['StateName'].to_dict()
    for col_name, col_dict in [('Breed1', breed), ('Breed2', breed), ('Color1', color),
                               ('Color2', color), ('Color3', color), ('State', state)]:
        df = _from_dict(df, col_name, mapping=col_dict)
    df[PET_IMAGE_COLUMN] = df['PetID'].apply(lambda x: _path_if_exists(x=x))
    df[LABEL_NAME] = df[LABEL_NAME].apply(_map_label)
    df = df[df[PET_IMAGE_COLUMN] != ""]
    df.drop(columns=['RescuerID', 'PetID'], inplace=True)
    return df


def _get_csv(csv: str) -> DataFrame:
    return read_csv(join(COMPETITION_FOLDER, csv))


def _from_dict(df: DataFrame, col: str, mapping: dict) -> DataFrame:
    df[col] = df[col].apply(lambda x: mapping.get(x, ''))
    return df


def _path_if_exists(x: str) -> str:
    img_name = f"{x}-1.jpg"
    path = join(IMAGE_FOLDER, img_name)
    if exists(path):
        return img_name
    return ""


def _map_label(i: int) -> str:
    mapping = {0: 'Same Day', 1: '1-7 Days', 2: '8-30 Days', 3: '31-90 Days', 4: 'Not adopted in 100 days'}
    return mapping[i]


CONTEXT = "Predicting speed of pet adoption from images, text description, and pet metadata"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = []
FEATURES = [
    CuratedFeature(raw_name=PET_IMAGE_COLUMN, feat_type=FeatureType.IMAGE),
    CuratedFeature(raw_name='Name', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='Description', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = join(COMPETITION_FOLDER, "train_images")
LOADING_FUNC = load_df
