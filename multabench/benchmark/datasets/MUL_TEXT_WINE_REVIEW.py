"""
Dataset Name: MUL_TEXT_WINE_REVIEW
====
Examples: 84123
====
URL: https://www.openml.org/search?type=data&id=46653
====
Target Variable: variety (object, 30 distinct): ['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc', 'Syrah', 'Rosé', 'Merlot']
====
Features:

country (object, 40 distinct, 0.0% missing): ['US', 'France', 'Italy', 'Portugal', 'Chile', 'Spain', 'Argentina', 'Austria', 'Australia', 'Germany']
description (object, 78995 distinct): ["Ripe plum, game, truffle, leather and menthol are some of the aromas you'll find on this earthy wine. The tightly wound palate offers dried black cherry, chopped sage, mint and roasted coffee bean alongside raspy tannins that leave a mouth-drying finish.", 'Seductively tart in lemon pith, cranberry and pomegranate, this refreshing, light-bodied quaff is infinitely enjoyable, both on its own or at the table. It continues to expand on the palate into an increasing array of fresh flavors, finishing in cherry and orange.', 'Fermented in an open stone lagar, giving great depth of color as well as rich, plummy concentration. The wine is ripe and bold, a plethora of licorice and bitter coffee to go with the rich red fruit. The tannins and acidity are finely in balance.', 'This is a very fragrant and floral rosé, reminiscent of peach gummy candies, orange blossom and orange-ginger tea. The palate is surprisingly lush, with round ripe-fruit flavors that are just barely lifted on the finish. Drink now.', 'Lots of personality in this Viognier. It has a meaty fleshiness, in addition to the peach pulp, orange and papaya fruit flavors, with fine acidity that makes it all feel clean and lively in the mouth. An interesting wine to pair with modern pan-Asian-fusion fare.', 'A deeply tawny gold color, this offers slightly oxidized, dusty peach and pear fruit, with a bit of residual sugar (20g/L) to round off the rough edges. Drink now through the end of 2017.', 'The 2008 Sineann Pinots are an elegant, refined group of wines, more expressive of Burgundian varietal character than any in memory, and the Resonance, as usual, is among the best. It has a vibrant purity to the fruit that rings true and long through mixed red berries, vivid acids, streaks of iron and earth, and sails on into a detailed and seamless finish.', 'This is a rich, robust Pinot Noir, defining the big, tannic style. It explodes in cherry, cola and pomegranate fruit flavors, with the added note of caramelized oak. You can drink this wine now, but it should hold in the bottle for 6–8 years, gradually mellowing. The vineyard is in the cooler southern part of the valley, near Carneros.', 'Rough and harsh in texture, this is a rustic, tannic blend of Tempranillo and Grenache that has overripe, burnt raisin flavors.', 'This is aged in 100% new American oak, putting tight berry and cassis flavors behind plenty of dried herb notes and astringent tannins. Fine for drinking with a thick steak, but not so much for sipping on its own.']
points (int64, 21 distinct): ['88', '87', '90', '86', '89', '91', '92', '85', '93', '84']
price (float64, 342 distinct, 6.6% missing): ['20.0', '15.0', '25.0', '30.0', '18.0', '40.0', '35.0', '12.0', '50.0', '10.0']
province (object, 361 distinct, 0.0% missing): ['California', 'Washington', 'Bordeaux', 'Oregon', 'Tuscany', 'Burgundy', 'Mendoza Province', 'Piedmont', 'New York', 'Alsace']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_WINE_REVIEW"
SLUG_BASE = "multabench-wine-review"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46653"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_FOOD_WINE_REVIEW)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
