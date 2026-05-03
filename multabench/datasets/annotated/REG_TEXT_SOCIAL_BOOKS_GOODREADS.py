from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType
from multabench.datasets.utils import load_csv

'''
Dataset Name: REG_SOCIAL_BOOKS_GOODREADS
====
Examples: 3967
====
URL: http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv
====
Description:
Goodreads: Datasets containing information about
books. The task is to predict the average rating of each
book.
====
Target Variable: Rating (float64, 214 distinct): ['4.0', '0.0', '3.0', '5.0', '3.5', '4.5', '3.67', '4.33', '3.75', '4.25']
====
Features:

Title (object, 3455 distinct): ['Autobiography', 'An Autobiography', 'The Autobiography', 'My Autobiography', ...]
Description (object, 2614 distinct): [...]
ISBN (object, 3080 distinct): ['0297792857', '386521472X', '0805463232', ...]
ISBN13 (object, 3038 distinct): [' ', '9781477581728', '9780854300563', ...]
PageCount (int64, 576 distinct): ['0', '320', '256', '288', '224', '304', ...]
FirstAuthor (object, 3211 distinct): ['Mark Twain', 'Benjamin Franklin', 'Anonymous', ...]
SecondAuthor (object, 841 distinct): [' ', 'Benjamin Franklin', 'Julia Watson', ...]
ThirdAuthor (object, 190 distinct): [' ', 'Anita Pacheco', 'Ryan Giggs', ...]
NumberofRatings (int64, 473 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '9']
NumberofReviews (object, 179 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Publisher (object, 1698 distinct): [' ', 'Hodder & Stoughton', 'Kessinger Publishing', ...]
PublishDate (object, 1896 distinct): [' ', '2009', 'March 1st 2007', 'January 1st 1992', ...]
Format (object, 35 distinct): ['Paperback', 'Hardcover', ' ', 'Kindle Edition', 'Unknown Binding', ...]
Language (object, 15 distinct): ['English', ' ', 'German', 'French', 'Russian', ...]
'''


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "goodreads.csv")


CONTEXT = "Books ratings"
TARGET = CuratedTarget(raw_name="Rating", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ["FileName", "ID"]
FEATURES = [CuratedFeature(raw_name="PublishDate", feat_type=FeatureType.DATE)]
LOADING_FUNC = load_df
