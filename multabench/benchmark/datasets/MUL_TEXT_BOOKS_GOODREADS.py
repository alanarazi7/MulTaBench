"""
Dataset Name: MUL_TEXT_BOOKS_GOODREADS
====
Examples: 3967
====
URL: https://www.kaggle.com/datasets/http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv
====
Target Variable: Rating (object, 3 distinct): ['Quantile 33-67%', 'Quantile 0-33%', 'Quantile 67-100%']
====
Features:

Title (object, 3455 distinct): ['Autobiography', 'An Autobiography', 'The Autobiography', 'My Autobiography', 'Autobiographies', 'The Test: My Autobiography', 'The Way The Wind Blows: An Autobiography', 'An Autobiography of Davy Crockett', 'D-Day Survivor: An Autobiography', 'The Last Maharani of Gwalior: An Autobiography']
Description (object, 2614 distinct, 0.0% missing): [' ', 'This book was converted from its physical edition to the digital format by a community of volunteers. You may find it for free on the web. Purchase of the Kindle edition includes wireless delivery.', 'Kessinger Publishing is the place to find hundreds of thousands of rare and hard-to-find books with something of interest for everyone', 'This is a reproduction of a book published before 1923. This book may have occasional imperfections such as missing or blurred pages, poor pictures, errant marks, etc. that were either part of the original artifact, or were introduced by the scanning process. We believe this work is culturally important, and despite the imperfections, have elected to bring it back into print as part of our continuing commitment to the preservation of printed works worldwide. We appreciate your understanding of the imperfections in the preservation process, and hope you enjoy this valuable book.', 'Many of the earliest books, particularly those dating back to the 1900s and before, are now extremely scarce and increasingly expensive. We are republishing these classic works in affordable, high quality, modern editions, using the original text and artwork.', 'This is a pre-1923 historical reproduction that was curated for quality. Quality assurance was conducted on each of these books in an attempt to remove books with imperfections introduced by the digitization process. Though we have made best efforts - the books may have occasional errors that do not impede the reading experience. We believe this work is culturally important and have elected to bring the book back into print as part of our continuing commitment to the preservation of printed works worldwide.', "This is an EXACT reproduction of a book published before 1923. This IS NOT an OCR'd book with strange characters, introduced typographical errors, and jumbled words. This book may have occasional imperfections such as missing or blurred pages, poor pictures, errant marks, etc. that were either part of the original artifact, or were introduced by the scanning process. We believe this work is culturally important, and despite the imperfections, have elected to bring it back into print as part of our continuing commitment to the preservation of printed works worldwide. We appreciate your understanding of the imperfections in the preservation process, and hope you enjoy this valuable book.", 'This work has been selected by scholars as being culturally important, and is part of the knowledge base of civilization as we know it. This work was reproduced from the original artifact, and remains as true to the original work as possible. Therefore, you will see the original copyright references, library stamps (as most of these works have been housed in our most important libraries around the world), and other notations in the work. This work is in the public domain in the United States of America, and possibly other nations. Within the United States, you may freely copy and distribute this work, as no entity (individual or corporate) has a copyright on the body of the work.As a reproduction of a historical artifact, this work may contain missing or blurred pages, poor pictures, errant marks, etc. Scholars believe, and we concur, that this work is important enough to be preserved, reproduced, and made generally available to the public. We appreciate your support of the preservation process, and thank you for being an important part of keeping this knowledge alive and relevant.', "This scarce antiquarian book is a selection from Kessinger Publishing's Legacy Reprint Series. Due to its age, it may contain imperfections such as marks, notations, marginalia and flawed pages. Because we believe this work is culturally important, we have made it available as part of our commitment to protecting, preserving, and promoting the world's literature. Kessinger Publishing is the place to find hundreds of thousands of rare and hard-to-find books with something of interest for everyone", "Barrie McDermott has been at the very top of British rugby league for more than a decade, starring for Oldham, Wigan and Leeds, and earning caps for England, Ireland, and Great Britain. But what is not widely known is the fact that McDermott has achieved all this despite the handicap of having lost an eye in a shooting accident when he was just 15. He has appeared before the Rugby Football League's disciplinary committee more than a dozen times, missing over 40 matches through suspension. This outspoken and fascinating autobiography of one of rugby's hardest men lifts the lid on one of the most remarkable careers in British sport."]
ISBN (object, 3080 distinct, 16.2% missing): ['1782798048', '0951936905', '1847712924', '0820479446', '0938459031', '1434353419', '190431726X', '087023756X', '0681416513', '9711003112']
ISBN13 (object, 3038 distinct): [' ', '9780882862118', '9780706438079', '9780404200879', '9780752888378', '9780805463231', '9781892446077', '9780340708521', '9780930350819', '9780859553087']
PageCount (int64, 576 distinct): ['0', '320', '256', '288', '224', '304', '192', '352', '336', '240']
FirstAuthor (object, 3211 distinct): ['Mark Twain', 'Benjamin Franklin', 'Booker T. Washington', 'Anonymous', 'Jon E. Lewis', "Seán O'Casey", 'Theodore Roosevelt', 'Dr. Block', 'Leonard Woolf', 'Sidonie Smith']
SecondAuthor (object, 841 distinct): [' ', 'Benjamin Franklin', 'Julia   Watson', 'Tom Carter', 'Maureen Lipman', 'Anthony Bozza', 'Harriet E. Smith', 'Wilton Earle', 'David Dalton', 'S.T. Joshi']
ThirdAuthor (object, 190 distinct): [' ', 'Grover Gardner', 'Anita Pacheco', 'Eleanor Zelliot', 'David E. Schultz', 'Ryan Giggs', 'Tomi Jill Folk', 'Charles H. Red Corn', 'Lily Chia Brissman', 'Billie Stafford']
NumberofRatings (int64, 473 distinct): ['1', '0', '2', '3', '4', '5', '6', '8', '7', '9']
NumberofReviews (object, 179 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Publisher (object, 1698 distinct): [' ', 'Hodder & Stoughton', 'Kessinger Publishing', 'Createspace', 'iUniverse', 'Simon & Schuster', 'Orion Publishing', 'Headline Book Publishing', 'Oxford University Pres', 'Not Avail']
PublishDate (datetime64[ns], 534 distinct, 63.1% missing): ['2005-01-01 00:00:00', '2007-03-01 00:00:00', '1992-01-01 00:00:00', '1995-01-01 00:00:00', '1990-01-01 00:00:00', '2012-06-01 00:00:00', '2007-10-01 00:00:00', '1993-01-01 00:00:00', '2004-09-01 00:00:00', '2005-11-01 00:00:00']
Format (object, 35 distinct): ['Paperback', 'Hardcover', ' ', 'Kindle Edition', 'Unknown Binding', 'ebook', 'Nook', 'Library Binding', 'Mass Market Paperback', 'Audio CD']
Language (object, 15 distinct, 0.5% missing): ['English', ' ', 'German', 'French', 'Russian', 'Hungarian', 'Serbian', 'Chinese', 'Swedish', 'Korean']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import bin_target, save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_BOOKS_GOODREADS"
SLUG_BASE = "multabench-full-books-goodreads"
KAGGLE_SOURCE = "http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv"
TARGET_BINS = 3


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_SOCIAL_BOOKS_GOODREADS)
    y = bin_target(dataset.y, n_bins=TARGET_BINS)
    df = pd.concat([dataset.x, y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description="Curate MUL_TEXT_BOOKS_GOODREADS for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
