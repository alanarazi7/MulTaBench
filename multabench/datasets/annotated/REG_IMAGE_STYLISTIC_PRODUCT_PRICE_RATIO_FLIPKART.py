import os
from os.path import exists, join

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: kuchhbhi/stylish-product-image-dataset/
====
Examples: 60900
====
URL: https://www.kaggle.com/kuchhbhi/stylish-product-image-dataset
====
Description:
title: Stylish Product Image Dataset
subtitle: 65K Records of Fashion Product Image
keywords: ['clothing and accessories', 'business', 'recommender systems', 'tabular', 'image']
licenses: [{'name': 'CC0-1.0'}]
description: # Context:
The idea came to my mind to scrap this data. I was working on an e-commerce project **[Fashion Product Recommendation](https://fashnkart.herokuapp.com/)** (an end-to-end project). In this project, upload any fashion image and it will show the 10 closest recommendations.

![](https://user-images.githubusercontent.com/40932902/169657090-20d3342d-d472-48e3-bc34-8a9686b09961.png)

![](https://user-images.githubusercontent.com/40932902/169657035-870bb803-f985-482a-ac16-789d0fcf2a2b.png)

![](https://user-images.githubusercontent.com/40932902/169013855-099838d6-8612-45ce-8961-28ccf44f81f7.png)

I completed my project on this  [image dataset ](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).
The problem I faced while deploying on the Heroku server. Due to the large project file size,  I was unable to deploy as **Heroku** offers limited memory space for a free account.

As currently, I am only familiar with Heroku. Learning AWS for big projects.
So, I decided to scrap my own image dataset with much more information that can help me to transform this project to the next level.
Scraped this data from **flipkart.com**(e-commerce website) in two formats Image and textual data in tabular format.

# About this Dataset:
This dataset contains **65k images (400x450 pixel)**) of fashion/style products and accessories like clothing, footwear, accessories, and many more.
There is a **CSV** file also mapped with the image name and the id column in tabular data.
The name of the image is in a unique numerical format like 1.png, 62299.png
Image name and Id columns are the same. So, suppose you want to find the details of any image then you can find them using the image name id, go to the Id column in the csv file and that id rows will be the details of the image.
You can find the notebook in the code section which I used to scrap this data.

Columns of CSV Dataset:
1. **id** : Unique id same as the image name
2. **brand**: Brand name of the product
3. **title**: Title of the product
4. **sold_price**: selling price of the product
5. **actual_price**: Actual price of the product
6. **url** : unique URL of every product
7. **img**: Image URL

How did helped me this dataset:
1. I trained my CNN model using the image data, that's the only use of the image dataset.
2. In my front-end page of the project to display results, I used Image URL and displayed after extracting from the web. This helped me to not upload the image dataset with the project on the server and this saved huge memory space.
3.  Using the **url** displaying live **price** and** ratings** from the Flipkart website.
4. And there is a Buy button mapped with the **url**  you will be redirected to the original product page and buy it from there.
after using this dataset I changed my project name from **Fashion Product Recommender** to **Flipkart Fashion Product Recommender**.  😄😄😄

Still, the memory problem was not resolved as the model trained file was above 500MB on the complete dataset. So I tried on multiple sets and finally, I deployed after training on 1000 images only. In the future, I will try on another platform to deploy the complete project.
 I learned many new things while working on this dataset.

## Your Job:
1. You can use this dataset in your deep learning projects, go and try to create interesting projects.
2.  You can use CSV data in your Machine Learning projects, first you need to do feature construction from the title columns as there is much information hidden and some data cleaning required.
3. There is two complete records missing in csv data, your job is to find the missing data with the help of image dataset and fill as per your knowledge.

### This is a huge dataset in terms of records as well as memory size. To download this dataset you need high internet speed.
To download the same dataset in **[small size less than 500mb](https://www.kaggle.com/datasets/kuchhbhi/flipkart-fashion-products-65k-dataset)** you can find it here,
everything is the same as this dataset only I reduced the pixel of the image from 400x450px to ** 65x80pixels**.

### Pls, Rate this work
## Support with Upvote... that encourages me to research more.
## Share your feedback, reviews, and suggestions if any.
#Thanks!!
====
Target Variable: price_ratio (float64, 14349 distinct): ['0.2993', '0.4995', '0.3994', '0.2496', '0.3329', '0.2392', '0.4494', '0.3493', '0.2492', '0.1992']
====
Features:

id (object, 60900 distinct): ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '10.png']
brand (object, 7208 distinct, 6.5% missing): ['PUMA', 'PETER ENGLAND', 'Allen Solly', 'HIGHLANDER', 'CAMPUS', 'METRONAUT', 'ADIDAS', 'Roadster', 'ASIAN', 'ARROW']
title (object, 15510 distinct): ['Men Cargos', 'Round Neck Women Blouse', 'Loafers For Men', 'Solid Men Three Fourths', 'Embroidered Semi Stitched Lehenga Choli', 'Sneakers For Men', 'Bellies For Women', 'Cotton Solid Patiala', 'Boots For Men', 'Unstitched Cotton Polyester Blend Shirt Fabric Printed']
actual_price (float64, 1360 distinct): ['999.0', '1999.0', '1499.0', '1299.0', '1599.0', '499.0', '2999.0', '799.0', '899.0', '2499.0']
'''

IMAGE_FEATURE_NAME = "id"


def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "Data - Copy.csv")
    df[IMAGE_FEATURE_NAME] = df[IMAGE_FEATURE_NAME].apply(lambda img: get_style_img(img, dir_path))
    df['sold_price'] = df['sold_price'].apply(_parse_indian_price)
    df['actual_price'] = df['actual_price'].apply(_parse_indian_price)
    df['price_ratio'] = df['sold_price'] / df['actual_price']
    return df


def get_style_img(img: str, dir_path: str) -> str:
    img_folder = join(dir_path, IMAGE_FOLDER)
    img_filename = f"{img}.png"
    if not exists(join(img_folder, img_filename)):
        return None
    return img_filename


def _parse_indian_price(p: str | float) -> float:
    # ₹939, ₹8,599        
    if isinstance(p, float):
        return p
    assert isinstance(p, str)
    return float(p.replace('₹', '').replace(',', ''))                           


TARGET = CuratedTarget(raw_name='price_ratio', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['img', 'url', 'sold_price']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='brand', feat_type=FeatureType.TEXT),
            CuratedFeature(raw_name='title', feat_type=FeatureType.TEXT),
]
IMAGE_FOLDER = "Fashion_Products_Image/Flipkart"
LOADING_FUNC = load_df
