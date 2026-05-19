"""
Dataset Name: MUL_TEXT_PRODUCT_SENTIMENT
====
Examples: 5091
====
URL: https://www.openml.org/search?type=data&id=46651
====
Target Variable: Sentiment (object, 4 distinct): ['Positive', 'No Sentiment', 'Negative', 'Cannot Say']
====
Features:

Product_Description (object, 5084 distinct): ['RT @mention Marissa Mayer: Google Will Connect the Digital &amp; Physical Worlds Through Mobile - {link} #sxsw', 'RT @mention \x89÷¼ GO BEYOND BORDERS! \x89÷_ {link} \x89ã_ #edchat #musedchat #sxsw #sxswi #classical #newTwitter', "RT @mention RT @mention It's not a rumor: Apple is opening up a temporary store in downtown Austin for #SXSW and the iPad 2 launch {link}", 'RT @mention Google to Launch Major New Social Network Called Circles, Possibly Today {link} #sxsw', 'RT @mention Marissa Mayer: Google Will Connect the Digital &amp; Physical Worlds Through Mobile - {link} #SXSW', 'Win free ipad 2 from webdoc.com #sxsw RT', "RT @mention \x89÷¼ Happy Woman's Day! Make love, not fuss! \x89÷_ {link} \x89ã_ #edchat #musedchat #sxsw #sxswi #classical #newTwitter", '@mention you can check out {link} for other #SXSW iPad apps too.', "Mayer says it makes sense to condense Google's location products &amp; features now that experiments show which ones are successful #SxSW #SUxSW", "It's on, @mention just walked in to The Industry Party by #GSDM &amp; #Google Austin, TX. #SXSW"]
Product_Type (int64, 10 distinct): ['9', '6', '2', '7', '3', '5', '8', '1', '0', '4']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_PRODUCT_SENTIMENT"
SLUG_BASE = "multabench-product-sentiment"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46651"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
