import json

from pandas import DataFrame

from multabench.datasets.utils import load_csv
from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType


'''
Dataset Name: nsmlehq/tokopedia-products-2025/
====
Examples: 1200
====
URL: https://www.kaggle.com/nsmlehq/tokopedia-products-2025
====
Description: 
About Dataset
Tokopedia Products Dataset (2025)
📖 Overview
This dataset contains a structured collection of product listings scraped from Tokopedia, Indonesia’s largest e-commerce platform. It captures detailed product, pricing, seller, and transactional metadata, providing a realistic snapshot of online retail activity in 2025.

The dataset is designed to support data analysis, machine learning, recommendation systems, and e-commerce research, particularly in areas such as pricing strategy, consumer behavior, product categorization, and seller performance evaluation.
All records are stored in csv and json format, with special handling for multi-value fields such as image lists.

📊 Dataset Specifications
Metric	Details
Total Product	1,200 samples
Collection Date	December 2025
Source	Tokopedia (public product pages)
Language	Indonesian, including emojis
File Format	CSV (Comma-separated Values) and JSON (JavaScript Object Notation)
Encoding	utf-8
📁 Dataset Structure
The dataset is delivered as a ready-to-use bundle consisting of structured metadata files and associated image assets. All components are organized to support seamless integration into data science workflows, machine learning pipelines, and analytical environments.

The dataset includes:

Structured tabular data (products.csv) for fast analysis and modeling.
Hierarchical JSON data (products.json) for flexible programmatic access.
Image assets (images/, thumbnails/) aligned with metadata references.
dataset/
│
├── products.csv
├── products.json
│
├── images/
│   ├── 00009699369d4112b505204cff43f97f~.jpeg
│   ├── 0005bd35-a5af-4e2f-9cc2-6c9992277ed9.jpg
│   └── ...
│
└── thumbnails/
    ├── 00009699369d4112b505204cff43f97f~.jpeg
    ├── 0005bd35-a5af-4e2f-9cc2-6c9992277ed9.jpg
    └── ...

📄 Tabular Column (products.csv)
Column Name	Description
id	Unique product ID
name	Product name or title
description	Full product description text
min_order	Minimum order quantity
max_order	Maximum order quantity
weight	Product weight
weight_unit	Weight unit (e.g., Gram)
condition	Product condition (e.g., New, Used)
status	Product availability status
applink	Deep link to product in Tokopedia mobile app
url	Product web URL
thumbnail	Thumbnail image path
images	List of product image paths
price	Current product price (numeric)
price_text	Formatted price string
price_slash_text	Original price before discount
discount_percent	Discount percentage
shop_id	Seller unique identifier
shop_url	Seller page URL
shop_city	Seller city
shop_name	Seller or store name
shop_applink	Seller deep link
shop_reputation	Seller reputation score or badge
warehouse_id	Warehouse identifier
parent_product_id	Parent product reference ID
category_id	Product category ID
category_name	Product category name
category_url	Category page URL
transaction_success	Number of successful transactions
transaction_reject	Number of rejected transactions
sold_count	Total number of units sold
sold_count_text	Human-readable sold count (e.g., "1 rb+")
view_count	Total number of product views
review_count	Number of customer reviews
talk_count	Number of customer discussions
rating	Average product rating (1–5 scale)
📄 Structured Object Fields (products.json)
Field Name	Type	Description
id	Integer	Unique product identifier
name	String	Full product name or title
description	String	Detailed product description
minOrder	Integer	Minimum order quantity
maxOrder	Integer	Maximum order quantity
weight	Integer	Product weight value
weightUnit	String	Unit of weight (e.g., Gram)
condition	String	Product condition (New / Used)
status	String	Product availability status
applink	String	Deep link to product in Tokopedia app
url	String	Product page URL
thumbnail	String	Thumbnail image paths
images	Array	Array of string product image paths
🎯 Intended Use Cases
This dataset is suitable for a wide range of applications, including but not limited to:

Product recommendation systems.
E-commerce analytics & demand forecasting.
Market basket and pricing analysis.
Multimodal learning (text + image + metadata).
Benchmarking ML/NLP models on real-world commercial data.
Product categorization and attribute extraction.
⚠️ Disclaimer
This dataset is intended for research and educational purposes only. All product names, trademarks, and brand references belong to their respective owners.

📜 License & Attribution
If you use this dataset in academic work, publications, or derivative datasets, please include proper attribution in the following format:

BibTeX
@misc{nsmle2025tokopediaproducts,
  title={Tokopedia Products Dataset (2025)},
  author={Pratama, Fiki},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/ds/9118061},
  doi={10.34740/kaggle/ds/9118061}
}
APA
Fiki Pratama. (2025). Tokopedia Products Dataset (2025) [Data set]. Kaggle. https://doi.org/10.34740/kaggle/ds/9118061
====
Target Variable: weight (int64, 225 distinct): ['1000', '500', '200', '100', '300', '400', '2000', '800', '150', '250']
====
Features:

name (object, 1198 distinct): ['(NEW LAUNCH) Soundcore Anker R50i NC Earbuds Adaptive Noise Canceling Headset Earphone Bluetooth 5.4 TWS Low Latency for Game Long Battery 4 Mics IP54 -A3959 - Black', 'LCD Display Module 3.5 inch TFT Touch Screen for Raspberry Pi', 'Stretch Film Plastik Wrapping 10cm 20cm 25cm 30cm 50cm 150m 200m 250m 300m PREMIUM - 20CM X 200M', 'Anker Laptop Power Bank Smart Digital Display Touch 25.000mAh Triple 100W USB-C ports 165W GaN Fast Charging PPS PD 3.0 UFCS Built-in and Retractable Cables iPhone, MacBook, Samsung Steam Deck DJI Flight\xa0safe\xa0-\xa0A1695 - Black', 'Link Khusus Spesial Order 2', '3.5" LCD TFT Touchscreen SPI Serial ILI9488 320*480 pixel for Arduino - Tnpa Toucscren', 'ecentio 750ml aluminium botol olahraga botol air multifungsi', 'Wifi Spy Camera model Powerbank - pantau live dari HP Android/iOS - 128 GB', 'Monster XKT30 Ear Clip Wireless Bluetooth 5.4 Headsets Super Bass Music Earphones Noise Reduction Sport - Beige', 'Payung Besar Jumbo Golf Otomatis Jari Fiber Premium Quality Anti Karat - Merah']
min_order (int64, 6 distinct): ['1', '2', '10', '6', '3', '5']
max_order (int64, 638 distinct): ['1', '2', '5', '8', '3', '0', '9', '17', '99999', '10']
condition (object, 2 distinct): ['New', 'Used']
status (object, 2 distinct): ['Active', 'Warehouse']
price (int64, 920 distinct): ['50000', '199000', '35000', '48000', '1499000', '59000', '129000', '49000', '1500000', '205000']
discount_percent (object, 83 distinct, 56.3% missing): ['50%', '53%', '30%', '3%', '20%', '40%', '67%', '15%', '52%', '59%']
shop_city (object, 57 distinct): ['Kota Administrasi Jakarta Barat', 'Kota Administrasi Jakarta Utara', 'Kab. Tangerang', 'Kota Administrasi Jakarta Pusat', 'Kota Tangerang', 'Kota Bandung', 'Kota Administrasi Jakarta Timur', 'Kab. Bekasi', 'Kota Bekasi', 'Kota Surabaya']
shop_name (object, 749 distinct): ['Xiaomi Indonesia', 'SOS SPORT ONLINE SHOP', 'Putra Group', 'N-Crypt ID', 'ecentio.indonesia', 'KHURS IOT', 'ISKU Tools Official Store', 'Toko Expert Komputer', 'MEGATRON.BIZ', 'Glad2Glow Official Store']
category_name (object, 387 distinct): ['Lainnya', 'VGA Card', 'Android OS', 'Mini PC', 'iOS', 'Monitor LED', 'TWS', 'Popok Sekali Pakai', 'Telepon Wireless', 'Harddisk External']
transaction_success (int64, 7 distinct): ['0', '1', '7', '2', '43', '8', '3']
transaction_reject (int64, 2 distinct): ['0', '1']
sold_count (int64, 952 distinct): ['1', '2', '3', '4', '8', '7', '9', '13', '15', '6']
sold_count_text (object, 53 distinct, 0.1% missing): ['10 rb+', '1 rb+', '2 rb+', '100 rb+', '100+', '50 rb+', '3 rb+', '5 rb+', '4 rb+', '250+']
review_count (int64, 843 distinct): ['1', '2', '4', '3', '6', '8', '5', '9', '11', '19']
rating (float64, 9 distinct): ['5.0', '4.9', '4.8', '4.7', '4.6', '4.5', '4.4', '4.0', '4.2']
image_cnt (int64, 43 distinct): ['5', '8', '10', '6', '7', '9', '1', '4', '11', '3']
img_0 (object, 1193 distinct): ['images/916afe16d5c349c4a3f4d18334c6281d~.jpeg', 'images/0b605728-0c10-467e-87a8-50ead45b6f1f.jpg', 'images/6a054128cf5c4b0aba2db8b673d72ccc~.jpeg', 'images/2c1f70c6623b496a913a01e513ae0486~.jpeg', 'images/a859f710-8ca0-461c-8de5-5af2ad2504a3.jpg', 'images/24f9226c-ffc5-41bb-9940-12902b8672c5.jpg', 'images/c0cb50fc8831485ab5fbeae3e23ccbd7~.jpeg', 'images/80839970-2fac-4fe3-bc3d-d50620f11879.png', 'images/9e216421ee164c6586201b82d6d64c2a~.jpeg', 'images/aae4198a-0a3c-495b-bb46-7fcbd761b726.png']
img_1 (object, 1126 distinct, 5.5% missing): ['images/c8b119fb23cf4a72baf32a731734e323~.jpeg', 'images/916afe16d5c349c4a3f4d18334c6281d~.jpeg', 'images/175170748_506dc68e-7c0a-4654-a6eb-02634b34793e_850_595.jpg', 'images/c1bf5598-3532-4f61-92bd-4b883d3f9fa1.jpg', 'images/2c1f70c6623b496a913a01e513ae0486~.jpeg', 'images/6a054128cf5c4b0aba2db8b673d72ccc~.jpeg', 'images/24f9226c-ffc5-41bb-9940-12902b8672c5.jpg', 'images/0b605728-0c10-467e-87a8-50ead45b6f1f.jpg', 'images/f48a3a0d4c6f44fcbde8dc22e77db991~.jpeg', 'images/1572f2e6a9d84b7e84dda472c1759849~.jpeg']
img_2 (object, 1094 distinct, 8.2% missing): ['images/701a96bb863842868138524db6528a3c~.jpeg', 'images/ac2c547537cf4127b21668ccc421b3eb~.jpeg', 'images/4467d9202a4342098eb33dd483a3bf1c~.jpeg', 'images/cf12a82b-fdb9-4325-ac12-ed8465402809.jpg', 'images/eaa335b7-fcfb-4236-b509-87b6ed4c1f42.jpg', 'images/82a94fe1-dceb-42dc-8be4-5526ef52a8f2.jpg', 'images/175170748_b09414d6-9158-4615-bdd9-7453b8dcb024_486_424.jpg', 'images/619c0846-0644-4e27-a54b-7a47004c43ac.jpg', 'images/954a8086c37742a981941d6b9a636fec~.jpeg', 'images/297a4cd0ac2e43078e9d0df6d06bbdcf~.jpeg']
'''

IMG_RAW = "images"

def load_df(dir_path: str) -> DataFrame:    
    df = load_csv(dir_path, "products.csv")
    handle_image_list(df)
    return df


def handle_image_list(df: DataFrame):
    # Image looks like: '["images/ca308f496d90410a81d67b58aa487df8~.jpeg","images/81df88c51f4e47f39ba3c1aa0816387f~.jpeg", ...]'
    # first, make it a list? it's a json/literal thing
    df[IMG_RAW] = df[IMG_RAW].apply(lambda i: json.loads(i))
    df['image_cnt'] = df['images'].apply(lambda i: len(i))
    df['img_0'] = df[IMG_RAW].apply(lambda i: i[0] if len(i) > 0 else None)
    df['img_1'] = df[IMG_RAW].apply(lambda i: i[1] if len(i) > 1 else None)
    df['img_2'] = df[IMG_RAW].apply(lambda i: i[2] if len(i) > 2 else None)
    for col in ['img_0', 'img_1', 'img_2']:
        df[col] = df[col].apply(_remove_images_without_valid_suffix)
    df.drop(columns=[IMG_RAW], inplace=True)
    return df

def _remove_images_without_valid_suffix(i: str | None) -> str | None:
    if not i:
        return None
    for suffix in [".jpg", ".jpeg", ".png"]:
        if i.endswith(suffix):
            return i
    return None

CONTEXT = ""
TARGET = CuratedTarget(raw_name="rating", task_type=SupervisedTask.MULTICLASS)
COLS_TO_DROP = [
    # IDs
    'id', 'category_id', 'parent_product_id', 'shop_id', 'warehouse_id',
    # URLs
    'category_url', 'shop_url', 'shop_applink', 'shop_reputation', 'url', 'applink', 'thumbnail',
    # Constants
    'view_count', 'talk_count', 'weight_unit',
    # Redundant
    'price_text', 'price_slash_text',
    # This is Indonese...
    'description']
FEATURES = [CuratedFeature(raw_name=f'img_{n}', feat_type=FeatureType.IMAGE) for n in range(3)]
IMAGE_FOLDER = ""
LOADING_FUNC = load_df
