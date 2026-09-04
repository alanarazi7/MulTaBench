"""
Dataset Name: REG_TEXT_WINE_POLISH_MARKET
====
Examples: 2247
====
URL: https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market
====
Target Variable: price (PLN) (float64, 420 distinct): ['89.0', '99.0', '59.0', '69.0', '79.0', '49.0', '119.0', '109.0', '55.0', '139.0']
====
Features:

name (object, 2236 distinct): ['Szampan Ruinart Rose', 'Chateau Mouton Rothschild Cru Classe', 'Brut Rose Billecart-Salmon', 'Paco & Lola Albariño Rías Baixas DO', 'Montelliana Prosecco Spumante Extra Dry Treviso DOC', 'Szampan Rose de Saignee 1er Cru Rene Geoffroy', 'Brunello di Montalcino Riserva Capanna', "Feudo Arancio Nero d'Avola Sicilia DOC", 'Feudo Arancio Grillo Sicilia DOC', 'Chateau Musar']
country (object, 25 distinct, 0.4% missing): ['Italy', 'France', 'Spain', 'Austria', 'Poland', 'Portugal', 'Germany', 'Chile', 'Argentina', 'United States of America']
region (object, 136 distinct, 7.5% missing): ['Tuscany', 'Burgundy', 'Veneto', 'Champagne', 'Trentino Alto Adige', 'Sicily', 'Puglia', 'Rhône Valley', 'Catalonia', 'Piedmont']
appellation (object, 342 distinct, 27.4% missing): ['Champagne AOC', 'Rioja DOC', 'Toscana IGT', 'Vin de France', 'Alsace AOC', 'Ribera del Duero DO', 'Kamptal DAC', 'Terre Siciliane IGT', 'Venezia Gulia IGT', 'Porto DOC']
vineyard (object, 281 distinct, 45.3% missing): ['Cantine San Marzano', 'St. Michael-Eppan', 'Maison Michel Chapoutier', 'Auer', 'CAVIT Cantina Viticoltori del Trentino', 'Allegrini', 'Dr. Loosen', 'Niepoort', 'Aresti Chile Wines Limitada', 'Winnice Jaworek']
vintage (float64, 22 distinct, 12.1% missing): ['2021.0', '2020.0', '2019.0', '2018.0', '2022.0', '2017.0', '2016.0', '2015.0', '2014.0', '2013.0']
volume (liters) (float64, 11 distinct, 3.0% missing): ['0.75', '1.5', '0.375', '0.5', '1.0', '3.0', '5.0', '0.7', '0.2', '0.735']
alcohol (%) (float64, 39 distinct, 23.3% missing): ['13.5', '13.0', '12.5', '14.0', '12.0', '14.5', '11.5', '11.0', '15.0', '10.5']
serving temperature (C) (object, 22 distinct, 20.9% missing): ['16-18', '16', '10-12', '15', '12', '18', '10', '8', '8-10', '14']
color (object, 4 distinct, 11.1% missing): ['red', 'white', 'rose', 'orange']
kind (object, 4 distinct, 88.7% missing): ['sparkling', 'champagne', 'port', 'sherry']
taste (object, 4 distinct, 2.3% missing): ['dry', 'semi-dry', 'semi-sweet', 'sweet']
style (object, 3 distinct, 19.5% missing): ['average', 'full', 'light']
medals (object, 45 distinct, 92.8% missing): ['Mundus Vini Gold', 'Decanter Gold', 'Tre Bicchieri Gambero Rosso', 'Due Bicchieri Gambero Rosso', 'Decanter Silver', 'Decanter Bronze', 'Mundus Vini Silver', 'Berliner Gold', 'IWC Silver', 'IWC Gold']
wegan (bool, 2 distinct): ['0', '1']
natural (bool, 2 distinct): ['0', '1']
punctation (float64, 26 distinct, 84.6% missing): ['90.0', '93.0', '92.0', '91.0', '94.0', '95.0', '96.0', '91.5', '92.5', '93.5']
grapes (object, 649 distinct, 2.3% missing): ['Chardonnay', 'Pinot Noir', 'Riesling', 'Tempranillo', 'Sauvignon Blanc', 'Cabernet Sauvignon', 'Sangiovese', 'Primitivo', 'Gewürztraminer', 'Malbec']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_WINE_POLISH_MARKET"
SLUG_BASE = "multabench-full-wine-polish-market"
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market"


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(KaggleDatasetID.REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate REG_TEXT_WINE_POLISH_MARKET for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
