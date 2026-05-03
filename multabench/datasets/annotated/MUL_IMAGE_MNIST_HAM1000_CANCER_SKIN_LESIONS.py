import os
from os.path import join

import pandas as pd
from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

LABEL_NAME = "dx_type"
IMAGE_FEATURE_NAME = "image_id"

'''
Dataset Name: kmader/skin-cancer-mnist-ham10000/
====
Examples: 10015
====
URL: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
====
Description: 

Overview
Another more interesting than digit classification dataset to use to get biology and medicine students more excited about machine learning and image processing.

Original Data Source
Original Challenge: https://challenge2018.isic-archive.com
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
[1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368
[2] Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).

From Authors
Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small size and lack of diversity of available dataset of dermatoscopic images. We tackle this problem by releasing the HAM10000 ("Human Against Machine with 10000 training images") dataset. We collected dermatoscopic images from different populations, acquired and stored by different modalities. The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the HAM10000_metadata file.

The test set is not public, but the evaluation server remains running (see the challenge website). Any publications written using the HAM10000 data should be evaluated on the official test set hosted there, so that methods can be fairly compared.
====
Target Variable: dx_type (object, 4 distinct): ['histo', 'follow_up', 'consensus', 'confocal']
====
Features:

image_id (object, 10015 distinct): ['HAM10000_images_part_1/ISIC_0027419.jpg', 'HAM10000_images_part_1/ISIC_0025030.jpg', 'HAM10000_images_part_1/ISIC_0026769.jpg', 'HAM10000_images_part_1/ISIC_0025661.jpg', 'HAM10000_images_part_2/ISIC_0031633.jpg', 'HAM10000_images_part_1/ISIC_0027850.jpg', 'HAM10000_images_part_1/ISIC_0029176.jpg', 'HAM10000_images_part_1/ISIC_0029068.jpg', 'HAM10000_images_part_1/ISIC_0025837.jpg', 'HAM10000_images_part_1/ISIC_0025209.jpg']
age (float64, 18 distinct): ['45.0', '50.0', '55.0', '40.0', '60.0', '70.0', '35.0', '65.0', '75.0', '30.0']
sex (object, 3 distinct): ['male', 'female', 'unknown']
localization (object, 15 distinct): ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 'face', 'chest', 'foot', 'unknown', 'neck']
'''

def load_mnist_cancer_df(dir_path: str) -> DataFrame:
    csv_path = join(dir_path, 'HAM10000_metadata.csv')
    df = pd.read_csv(csv_path)
    img_name2path = {}
    for image_folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        images_path = join(dir_path, image_folder)
        for img_file in os.listdir(images_path):
            img_path = join(image_folder, img_file)
            img_without_suffix = img_file.split('.')[0]
            img_name2path[img_without_suffix] = img_path
    assert len(img_name2path) == df.shape[0], "Mismatch in number of images and metadata rows"
    df[IMAGE_FEATURE_NAME] = df['image_id'].map(img_name2path)
    return df


_LOC_GROUP = {
    'face': 'head_neck', 'neck': 'head_neck', 'scalp': 'head_neck', 'ear': 'head_neck',
    'back': 'trunk', 'trunk': 'trunk', 'abdomen': 'trunk', 'chest': 'trunk',
    'lower extremity': 'extremity', 'upper extremity': 'extremity',
    'foot': 'extremity', 'hand': 'extremity', 'acral': 'extremity',
}


def process_df(df: DataFrame) -> DataFrame:
    df['localization_group'] = df['localization'].map(_LOC_GROUP).fillna('other')
    return df


CONTEXT = "Classifying skin cancer lesions from MNIST images"
TARGET = CuratedTarget(raw_name=LABEL_NAME, task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = ["lesion_id"]  # dx kept so it can be used as target override
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_mnist_cancer_df
PROCESSING_FUNC = process_df
