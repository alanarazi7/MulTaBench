from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask

'''
Dataset Name: mercari_price_suggestion100K
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46660
====
Lennart:
This dataset comes from a consumer-to-consumer marketplace and includes listings with structured fields (e.g., item
condition, shipping) and high-cardinality textual inputs like product names, brand names, and descriptions. The goal is
to predict the item price. Hierarchical category fields (e.g., Women/Jewelry/Necklaces) are retained as textual inputs.
The dataset is ideal for evaluating semantic reasoning in text-rich, user-generated data.
====
Description: Predict the price of items sold in the online marketplace of Mercari based on
    information from the product page like name, description, free shipping availability, etc.
    This data originates from a 2017 Kaggle competition (https://www.kaggle.com/c/
    mercari-price-suggestion-challenge/), in which 1st place and 3rd place engineered
    dataset-specific text features such as customized bag-of-words and character N-grams, carefully
    tuned learning-rate/batch-size schedules, and specially ensembled models in a dataset-specific manner.

 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: log_price (float64, 415 distinct): ['2.3979', '2.5649', '2.7081', '2.8332', '2.3026', '2.1972', '2.7726', '3.0445', '2.0794', '3.2189']
====
Features:

name (object, 94033 distinct): ['Bundle', 'BUNDLE', 'Reserved', 'Lularoe TC leggings', 'Dress', 'Vans', 'Coach purse', 'Converse', 'Nike', 'Miss me jeans']
item_condition_id (uint8, 5 distinct): ['1', '3', '2', '4', '5']
category_name (object, 987 distinct, 0.4% missing): ['Women/Athletic Apparel/Pants, Tights, Leggings', 'Women/Tops & Blouses/T-Shirts', 'Beauty/Makeup/Face', 'Beauty/Makeup/Lips', 'Electronics/Video Games & Consoles/Games', 'Beauty/Makeup/Eyes', 'Electronics/Cell Phones & Accessories/Cases, Covers & Skins', 'Women/Underwear/Bras', 'Women/Tops & Blouses/Tank, Cami', 'Women/Tops & Blouses/Blouse']
brand_name (object, 2023 distinct, 42.5% missing): ['Nike', 'PINK', "Victoria's Secret", 'LuLaRoe', 'Apple', 'Lululemon', 'FOREVER 21', 'Nintendo', 'Michael Kors', 'American Eagle']
shipping (uint8, 2 distinct): ['0', '1']
item_description (object, 90584 distinct): ['No description yet', 'New', 'Brand new', 'Good condition', 'Great condition', 'Like new', 'Never worn', 'Excellent condition', 'NWT', 'Never used']
cat1 (object, 10 distinct, 0.4% missing): ['Women', 'Beauty', 'Kids', 'Electronics', 'Men', 'Home', 'Vintage & Collectibles', 'Other', 'Handmade', 'Sports & Outdoors']
cat2 (object, 113 distinct, 0.4% missing): ['Athletic Apparel', 'Makeup', 'Tops & Blouses', 'Shoes', 'Jewelry', 'Toys', 'Cell Phones & Accessories', "Women's Handbags", 'Dresses', "Women's Accessories"]
cat3 (object, 705 distinct, 0.4% missing): ['Pants, Tights, Leggings', 'Other', 'Face', 'T-Shirts', 'Shoes', 'Games', 'Lips', 'Athletic', 'Eyes', 'Cases, Covers & Skins']
'''

CONTEXT = "Mercari Online Marketplace Product Prices"
TARGET = CuratedTarget(raw_name="log_price", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = [
    # ID
    "train_id",
    # Target Leakage
    "price"]
FEATURES = []
