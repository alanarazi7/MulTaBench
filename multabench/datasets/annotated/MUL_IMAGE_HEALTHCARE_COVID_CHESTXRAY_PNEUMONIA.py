from os.path import join

from pandas import DataFrame, read_csv

from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: bachrr/covid-chest-xray/
====
Examples: ~929 (707 train, 222 test per BagOfTricks)
====
URL: https://www.kaggle.com/bachrr/covid-chest-xray
====
Bag Of Tricks Paper:
covid: Predict the pneumonia category (25 total) based on the chestxray images and metadata
including basic information of patients (age, sex, survival situation), clinical notes and so on.
This dataset originally stems from https://github.com/ieee8023/covid-chestxray-dataset.
We randomly split the original dataset at 3:1 ratio for new training and test set.
The license of the original dataset: Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0.

Description:
A database of COVID-19 cases with chest X-ray or CT images. It contains COVID-19 cases as well as
MERS, SARS, and ARDS cases.
====
Target Variable: finding (object, 25 distinct): ['COVID-19', 'Streptococcus', 'SARS', 'Pneumocystis', ...]
====
Features:

filename (object, ~929 distinct): chest X-ray image files
sex (object, 2 distinct, 12.3% missing): ['M', 'F']
age (float64, 54 distinct, 15.4% missing): ['70.0', '55.0', '50.0', '65.0', '60.0']
survival (object, 2 distinct, 67.0% missing): ['Y', 'N']
intubated (object, 2 distinct, 79.8% missing): ['Y', 'N']
intubation_present (object, 2 distinct, 78.3% missing): ['N', 'Y']
went_icu (object, 2 distinct, 90.0% missing): ['Y', 'N']
in_icu (object, 2 distinct, 98.0% missing): ['Y', 'N']
needed_supplemental_O2 (object, 2 distinct, 96.6% missing): ['Y', 'N']
extubated (object, 2 distinct, 93.4% missing): ['Y', 'N']
temperature (float64, 20 distinct, 90.0% missing): ['38.0', '39.0', '38.9']
pO2_saturation (float64, 18 distinct, 87.7% missing): ['97.0', '98.0', '96.0', '92.0']
view (object, 7 distinct): ['PA', 'AP', 'AP Supine', 'L', 'Axial', 'Coronal']
modality (object, 2 distinct): ['X-ray', 'CT']
location (object, 53 distinct, 31.9% missing): ['Italy', 'Spain', 'Mount Sinai Hospital, Toronto', ...]
clinical_notes (object, high cardinality): [clinical radiology notes ...]
'''

LABEL_NAME = "finding"
IMAGE_FEATURE_NAME = "filename"

_BAD_IMAGES = [
    'radiopaedia_org_covid-19-pneumonia-10_85902_3-dcm.nii.gz',
    "radiopaedia_org_covid-19-pneumonia-7_85703_0-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii.gz",
    "coronacases_org_001.nii.gz",
    "coronacases_org_002.nii.gz",
    "coronacases_org_003.nii.gz",
    "coronacases_org_004.nii.gz",
    "coronacases_org_005.nii.gz",
    "coronacases_org_006.nii.gz",
    "coronacases_org_007.nii.gz",
    "coronacases_org_008.nii.gz",
    "coronacases_org_009.nii.gz",
    "coronacases_org_010.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-4_85506_1-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-29_86490_1-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-29_86491_1-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-23_86359_0-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-10_85902_1-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-36_86526_0-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-27_86410_0-dcm.nii.gz",
    "radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.nii.gz",
]


def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "metadata.csv")
    df = read_csv(df_path)
    col_to_drop = [c for c in df.columns if 'unnamed' in c.lower()]
    df = df.drop(columns=col_to_drop)
    df = df[df[IMAGE_FEATURE_NAME].apply(_is_valid_img)]
    return df


def _is_valid_img(img_path: str) -> bool:
    for bad in _BAD_IMAGES:
        if bad in img_path:
            return False
    return True


CONTEXT = "Predicting pneumonia category from chest X-ray images and patient metadata"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = [
    "url", 'patientid', 'doi', 'other_notes', 'license', 'folder',
    'offset', 'date',
]
FEATURES = [
    CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
    CuratedFeature(raw_name='clinical_notes', feat_type=FeatureType.TEXT),
    CuratedFeature(raw_name='location', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "images"
LOADING_FUNC = load_df
