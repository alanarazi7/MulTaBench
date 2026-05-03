from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask
from multabench.datasets.utils import load_csv

'''
Dataset Name: ruthgn/beer-profile-and-ratings-data-set/
====
Examples: 3197
====
URL: https://www.kaggle.com/ruthgn/beer-profile-and-ratings-data-set
====
Description:
About Dataset
Data Set Information
This data set contains tasting profiles and consumer reviews for 3197 unique beers from 934 different breweries. It was created by integrating information from two existing data sets on Kaggle:

Beer Tasting Profiles Dataset
1.5 Million Beer Reviews
The purpose of the data integration is to create a new data set that contains comprehensive consumer review (appearance, aroma, palate, taste and overall review scores) for different brews, combined with their detailed tasting profiles—this is that data set.

In the future it might be possible to go through and scrape more detailed information for each brew, such as the brewery's country of origin or brew recipe information such as its original gravity vs. final gravity.

Contents
The main data set (beer_profile_and_ratings.csv) contains the following columns:

1 - Name: Beer name (label)
2 - Style: Beer Style
3 - Brewery: Brewery name
4 - Beer Name (Full): Complete beer name (Brewery + Brew Name) -- unique identifier for each beer
5 - Description: Notes on the beer if available
6 - ABV: Alcohol content of beer (% by volume)
7 - Min IBU: The minimum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was
8 - Max IBU: The maximum IBU value each beer can possess. IBU was not a value available for each beer, but the IBU range for each style was

The next eleven columns represent the tasting profile features of the beer, and are defined by word counts found in up to 25 reviews of each beer. The assumption is that people writing reviews are more than likely describing what they do experience rather than what they do not. (Refer to the file Beer Descriptors Simplified to see the list of words that are used to calculate the values contained in each of the feature columns below)

(Mouthfeel)
9 - Astringency
10 - Body
11 - Alcohol

(Taste)
12 - Bitter
13 - Sweet
14 - Sour
15 - Salty

(Flavor And Aroma)
16 - Fruits
17 - Hoppy
18 - Spices
19 - Malty

The last six columns contain information from beer reviews--they include the number (count) of consumer/user reviews, the average overall rating score, and the average rating scores for the aroma, appearance, palate, and taste of each individual beer.

20 - review _ aroma
21 - review _ appearance
22 - review _ palate
23 - review _ taste
24 - review _ overall
25 - number _ of _ reviews

The next two files (Brewery Name Fuzzy Match List.csv and Beer Name Fuzzy Match List.csv) only contain lists of breweries and beers that are found in both source datasets--and consequently, included in this dataset. To see the data integration process in more details, check out this notebook.

The last file (Beer Descriptors Simplified) contains list of words that are used to calculate the values contained in the tasting profile feature columns.

Acknowledgements
Source: BeerAdvocate
Credits:

Beer Tasting Profiles Dataset by sp1222.
1.5 Million Beer Reviews by Tanya Cashorali (uploaded by Datadoume).
Context
As previously mentioned, this data set was created to put existing consumer ratings data for different brews and their detailed tasting profiles in one place. Possible uses for the dataset include:

Analyzing the properties that make a highly-rated beer
Clustering and building a beer recommendation system based on similarities
Classifying different beer styles based on tasting profile information
Predicting a brew's alcohol content (ABV) using known characteristics
A lot more!
This one is for all the beer lovers out there—cheers!
====
Target Variable: review_overall (float64, 2325 distinct): ['4.0', '3.5', '3.75', '4.5', '3.6667', '3.0', '4.25', '3.875', '4.125', '3.8333']
====
Features:

Name (object, 3066 distinct): ['Oktoberfest', 'Porter', 'Smoked Porter', 'Christmas Ale', 'India Pale Ale', 'Brown Ale', 'IPA', 'Winter Ale', 'Nut Brown Ale', 'Grand Cru']
Style (object, 111 distinct): ['Lager - Adjunct', 'Lager - European Pale', 'Lambic - Fruit', 'Stout - Irish Dry', 'Wheat Beer - Hefeweizen', 'Dubbel', 'Bitter - English', 'Strong Ale - Belgian Dark', 'Pale Ale - English', 'Lager - Light']
Brewery (object, 934 distinct): ['Boston Beer Company (Samuel Adams)', 'Dogfish Head Brewery', 'Anheuser-Busch', 'Three Floyds Brewing Co. & Brewpub', 'Victory Brewing Company', 'Rogue Ales', 'Matt Brewing Company', "Short's Brewing Company", 'Russian River Brewing Company', 'Great Divide Brewing Company']
Beer Name (Full) (object, 3197 distinct): ['Alaskan Brewing Co. Alaskan Amber', 'Long Trail Brewing Co. Double Bag', 'Long Trail Brewing Co. Long Trail Ale']
Description (object, 1841 distinct): ['Notes:', 'Notes:21 IBU\t', 'Notes:Busch and Busch Light are both brewed with a blend of premium American-grown and imported hops and a combination of malt and corn to provide a pleasant balanced flavor.']
ABV (float64, 194 distinct): ['5.0', '5.5', '6.0', '8.0', '7.0', '6.5', '5.2', '4.5', '7.5', '4.8']
Min IBU (int64, 21 distinct): ['20', '15', '10', '25', '30', '18', '50', '0', '35', '5']
Max IBU (int64, 25 distinct): ['40', '30', '25', '50', '45', '35', '70', '60', '15', '100']
Astringency (int64, 66 distinct): ['13', '10', '15', '14', '12', '11', '9', '8', '7', '16']
Body (int64, 149 distinct): ['32', '38', '30', '34', '31', '36', '33', '40', '41', '35']
Alcohol (int64, 104 distinct): ['7', '6', '8', '5', '9', '4', '10', '11', '3', '2']
Bitter (int64, 133 distinct): ['25', '24', '18', '16', '17', '19', '35', '10', '22', '20']
Sweet (int64, 183 distinct): ['52', '47', '65', '48', '62', '71', '57', '54', '33', '55']
Sour (int64, 192 distinct): ['10', '9', '11', '13', '4', '8', '7', '6', '5', '14']
Salty (int64, 21 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Fruits (int64, 149 distinct): ['6', '5', '9', '13', '3', '10', '7', '4', '18', '8']
Hoppy (int64, 153 distinct): ['16', '20', '14', '25', '17', '24', '15', '19', '23', '26']
Spices (int64, 138 distinct): ['4', '3', '2', '5', '7', '6', '0', '1', '8', '9']
Malty (int64, 196 distinct): ['47', '80', '97', '74', '64', '61', '100', '72', '62', '53']
review_aroma (float64, 2326 distinct): ['3.5', '4.0', '3.75', '3.6667', '3.25', '3.0', '3.8333', '3.625', '4.5', '4.25']
review_appearance (float64, 2257 distinct): ['4.0', '3.5', '3.75', '3.0', '3.6667', '3.8333', '3.875', '4.25', '4.1667', '3.9']
review_palate (float64, 2324 distinct): ['4.0', '3.5', '3.75', '3.0', '3.8333', '3.6667', '4.5', '3.3333', '3.25', '4.1667']
review_taste (float64, 2356 distinct): ['4.0', '3.5', '3.75', '4.5', '3.0', '4.25', '3.8333', '3.875', '3.625', '2.5']
number_of_reviews (int64, 811 distinct): ['2', '3', '1', '4', '5', '6', '12', '7', '8', '13']
'''


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "beer_profile_and_ratings.csv")


TARGET = CuratedTarget(raw_name="review_overall", task_type=SupervisedTask.REGRESSION)
LOADING_FUNC = load_df
