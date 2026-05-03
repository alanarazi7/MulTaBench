import math
from os.path import join

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.utils.file_downloading import download_url_image_column
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: amankumar20d/amazon-best-seller-all-departments-us/
====
Examples: 3479
====
URL: https://www.kaggle.com/amankumar20d/amazon-best-seller-all-departments-us
====
Description:
title: Amazon_Best_Seller_All_Departments_US
subtitle:
keywords: ['business', 'tabular', 'image', 'text']
licenses: [{'name': 'MIT'}]
description: **Amazon Best Sellers Across Departments**
This dataset contains comprehensive information about the top-selling products from various departments on Amazon in the United States. The data was collected using the RapidAPI Amazon Online Data API, ensuring real-time accuracy and relevance. Each record represents a best-selling product, enriched with relevant details like department, product ranking, and additional metadata.


**Usage Examples**
- Market Research: Analyze top-selling products in various categories.
- Trend Analysis: Identify popular product trends across Amazon departments.
- Price Comparison: Compare pricing and customer feedback for top products.
- Recommendation Systems: Train machine learning models to recommend products.



**Dataset Size**
Total Departments: 40
Total Records: ~80 records per department (40 departments × 2 pages).



**Fields Included:**
product_title: Name of the product.
product_link: URL to the product page on Amazon.
rank: Product's rank within its category.
price: Price of the product (if available).
ratings: Number of customer ratings.
reviews: Total number of customer reviews.
department: The department/category the product belongs to.
page: The page number from which the data was fetched.
====
Target Variable: star_rating (float64, 35 distinct): ['4.7', '4.6', '4.8', '4.5', '4.4', '4.3', '4.2', '4.1', '4.9', '5.0']
====
Features:

num_ratings (object, 2953 distinct): ['777,049', 'Hat-Trick Games', '1', 'Intuit', '777049', '2', '6', '3', '7', '4']
photo_url (object, 3436 distinct): ['https___images-na.ssl-images-amazon.com_images_I_81bpKKv68-L._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_41rWhNPimsL._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_619mM1ncz4L._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_913C+MR3S5L._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_818i3AJdNdL._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_51D5YPHy0hL._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_710-budOI2L._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_516ej6vg9LL._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_61V5FRUgX8L._AC_UL300_SR300,200_.jpg', 'https___images-na.ssl-images-amazon.com_images_I_719ws-Z0IyL._AC_UL300_SR300,200_.jpg']
price (float64, 1140 distinct, 1.9% missing): ['9.99', '7.99', '19.99', '14.99', '8.99', '29.99', '39.99', '6.99', '50.0', '4.99']
rank (int64, 100 distinct): ['2', '12', '9', '7', '35', '33', '80', '50', '41', '42']
title (object, 2924 distinct, 12.4% missing): ['Amazon.com eGift Card (Instant Email or Text Delivery)', 'Amazon.com Download and Print at Home Gift Card', 'Amazon.com Gift Card in Various Gift Boxes', 'Apple Gift Card - App Store, iTunes, iPhone, iPad, AirPods, MacBook, accessories and more (eGift)', 'Amazon.com Gift Card in a Reveal (Various Designs)', 'All-new Amazon Kindle Paperwhite and Kindle Colorsoft Signature Edition Case, Lightweight and Water-Safe, Foldable Protective Cover - Fabric', 'Alpha Grillers Instant Read Meat Thermometer for Cooking Grilling and Griddle Accessories Kitchen Essentials - Waterproof Backlight & Calibration, Birthday Mens Gifts Valentines Day Gifts for Him', 'Amazon.com Gift Card in a Mini Envelope (General)', 'Blink Video Doorbell (newest model), Two-way audio, HD video, motion and chime app alerts and Alexa enabled — wired or wire-free (Black)', 'REALINN Under Sink Organizer and Storage, 2 Pack Pull Out Cabinet Organizer Slide Out Sink Shelf Cabinet Storage Shelves, Under Sink Storage for Kitchen Bathroom Cabinet']
department (object, 37 distinct): ['Amazon Devices & Accessories', 'Amazon Renewed', 'Appliances', 'Arts, Crafts & Sewing', 'Automotive', 'Sports & Outdoors', 'Baby', 'Beauty & Personal Care', 'Cell Phones & Accessories', 'Gift Cards']
page (int64, 2 distinct): ['1', '2']
'''

IMAGE_FEATURE_NAME = "photo_url"

def load_df(dir_path: str) -> DataFrame:
    df = load_csv(dir_path, "Amazon_Best_Seller.csv")
    img_folder = join(dir_path, IMAGE_FOLDER)
    df = download_url_image_column(df=df, img_folder=img_folder, img_col=IMAGE_FEATURE_NAME)
    df['price'] = df['price'].apply(_parse_price)
    return df


def _parse_price(price: str | float) -> float:
    if isinstance(price, float):
        return price
    assert isinstance(price, str)
    price = float(price.replace('$', '').replace(',', ''))
    return math.log1p(price)

def _parse_rating(rating: str | float) -> float | None:
    if isinstance(rating, float):
        return rating
    assert isinstance(rating, str)
    try:
        return float(rating)
    except ValueError:
        return None

CONTEXT = ""
TARGET = CuratedTarget(raw_name='price', task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['asin', 'Unnamed: 0', 'url', 'title', 'department']
FEATURES = [CuratedFeature(raw_name=IMAGE_FEATURE_NAME, feat_type=FeatureType.IMAGE),
            CuratedFeature(raw_name='num_ratings', processing_func=_parse_rating, feat_type=FeatureType.NUMERIC),
]
IMAGE_FOLDER = "downloaded_amazon_images"
LOADING_FUNC = load_df
