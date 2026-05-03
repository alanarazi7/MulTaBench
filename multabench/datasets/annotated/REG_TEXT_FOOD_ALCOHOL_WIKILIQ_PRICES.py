from typing import Any
from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: limtis/wikiliq-dataset/spirits_data.csv
====
Examples: 12869
====
URL: https://www.kaggle.com/limtis/wikiliq-dataset/spirits_data.csv
====
Description:
Wikiliq - Alcohol dataset (May, 2022)
Liquors, Whiskey, Vodka, Wine, Beer, etc.

About Dataset
Parsed data from WikiliQ website.
Data is presented as-is. It wasn't cleaned, nor modified in any way.

There are three CSV files, containing information about wine, beer and liquors (Including whiskey, beer, vodka, liquors and spirits)

Common columns between these 3 files:
Name (of the product),
Brand,
Country,
Categories (Type of beer/wine/liquor)
ABV (Alcohol by volume)
Suggested Serving Temperature (in Fahrenheit)
Rating (Scale 0 to 5)
Rate count (Amount of reviews, the rating is based on)
Price (Sometimes might be incorrect, be careful)
Volume (Volume, the price is set for)
Description [of a product] (Mostly descriptions, but sometimes may be filled with contacts of a brand, or company's history)

There are additional columns in every file, but names speak for themselves.

====
Features:

Unnamed: 0 (int64, 12869 distinct): ['0', '8584', '8574', '8575', '8576', '8577', '8578', '8579', '8580', '8581']
Name (object, 12804 distinct): ['Hiram Walker Brandy Apricot', 'Leroux Polish Blackberry Flavored Brandy', 'Toschi Lemoncello', 'Tippy Cow Chocolate Rum', 'Whitmeyer's Texas Peach Whiskey', 'Tippy Cow Vanilla Soft Serve Rum', 'Belvoir Organic Elderflower Presse', 'Montezuma Blue Tequila', 'Margaritaville Coconut Tequila', 'Hiram Walker Kirschwasser']
Country (object, 105 distinct): ['United States', 'Mexico', 'Scotland', 'France', 'Canada', 'United Kingdom', 'Italy', 'Ireland', 'Netherlands', 'Japan']
Brand (object, 3971 distinct): ['DeKuyper Liqueur', 'Bacardi Rum', 'Smirnoff Vodka', 'Jack Daniels', 'Johnnie Walker', 'Gordon  MacPhail', 'Jose Cuervo', 'Pinnacle', 'Absolut', 'Highland Park']
Categories (object, 152 distinct): ['Scotch Whisky, Whiskey', 'Vodka', 'Bourbon, Whiskey', 'Liqueur', 'Ready-to-Drink', 'Rye Whiskey, Whiskey', 'Mezcal', 'American Whiskey, Whiskey', 'Rum', 'Brandy, Cognac']
Tasting Notes (object, 1623 distinct): ['Citrus', 'Smooth', 'Grapefruit', 'Fruity', 'Apple', 'Peach', 'Crisp, Smooth', 'Balanced, Dried Fruit, Rich, Round, Smoky, Smooth, Vanilla', 'Tropical Fruit', 'Mint']
ABV (object, 462 distinct): ['40%', '35%', '45%', '43%', '46%', '30%', '50%', '15%', '42%', '20%']
Base Ingredient (object, 48 distinct): ['Corn', 'Barley', 'Desert Plant', 'Fruit', 'Barley, Corn, Rye', 'Sugar Cane', 'Sugar', 'Wheat', 'Rye', 'Potato']
Years Aged (float64, 49 distinct): ['12.0', '10.0', '15.0', '5.0', '3.0', '2.0', '18.0', '4.0', '8.0', '21.0']
Rating (float64, 17 distinct): ['5.0', '4.8', '4.9', '2.5', '4.7', '4.6', '4.5', '4.4', '3.3', '4.1']
Rate Count (int64, 189 distinct): ['1', '2', '3', '4', '5', '6', '7', '10', '8', '9']
Price (object, 2484 distinct): ['$29.99', '$19.99', '$0.00', '$39.99', '$21.99', '$24.99', '$49.99', '$32.99', '$34.99', '$12.99']
Volume (object, 2 distinct): ['750ml', '1L']
Description (object, 9696 distinct): ['Burnett\'s Flavored Vodkas combine the quality of Burnett\'s Vodka with all-natural flavors to deliver a superior taste.', '15 pack of 25 oz. prefilled shots', 'SKYY vodka is the first quadruple distilled, triple filtered premium American vodka created in San Francisco in 1992.']
'''


def remove_currency(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return float(text.replace('$', ''))


def remove_percentage(text: Any) -> float:
    if not isinstance(text, str):
        return text
    return float(text.replace('%', ''))


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "spirits_data.csv")


TARGET = CuratedTarget(raw_name="Price", task_type=SupervisedTask.REGRESSION,
                       processing_func=remove_currency)
COLS_TO_DROP = ["Unnamed: 0"]
FEATURES = [CuratedFeature(raw_name="ABV", feat_type=FeatureType.NUMERIC,
                           processing_func=remove_percentage),]
LOADING_FUNC = load_df