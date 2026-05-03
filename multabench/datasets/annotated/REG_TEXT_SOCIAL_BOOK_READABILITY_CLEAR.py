from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask
from multabench.datasets.utils import load_csv

'''
Dataset Name: verracodeguacas/clear-corpus/CLEAR.csv
====
Examples: 4726
====
URL: https://www.kaggle.com/verracodeguacas/clear-corpus/CLEAR.csv
====
Description:
About Dataset
📚 CommonLit Ease of Readability (CLEAR) Corpus 📚

Dive into this comprehensive dataset, curated by CommonLit in collaboration with Georgia State University. With approximately 5,000 reading passages spanning from the 3rd to 12th grade levels, this resource is a treasure trove for researchers, educators, and data enthusiasts alike.

🔍 Key Features:

Unique readability scores for each passage.
Text excerpts covering over 250 years of literature across various genres.
Meta-data including publishing year, genre, and more.
Readability indices such as Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, and predictions from the Kaggle Readability Prize.
An invaluable resource for NLP, education, and literary analysis.
🔄 Continuously updated by CommonLit, ensuring fresh and accurate data.

Whether you're developing a new readability metric, analyzing literary trends, or training a machine learning model, the CLEAR Corpus is your go-to dataset! 🚀

====
Features:

ID (float64, 4724 distinct): ['400.0', '5460.0', '5477.0', '5476.0', '5475.0', '5474.0', '5473.0', '5472.0', '5471.0', '5470.0']
Last Changed (float64, 1 distinct): ['6.01']
Author (object, 2409 distinct): ['simple wiki', 'wikipedia', '?', 'USHistory.org', 'CommonLit Staff', 'wikijunior']
Title (object, 4658 distinct): ['Invention and Discovery', '?', 'Current History', 'LITTLE MISCHIEF', 'Bacteria', 'THE FAIRY GODMOTHERS']
Anthology (object, 290 distinct): ['CLD', 'Frontiers for Young Minds', 'African Storybook Level 3', 'African Storybook Level 4']
URL (object, 3688 distinct): ['https://www.africanstorybook.org/', 'https://www.africanstorybook.org/#']
Source (object, 19 distinct): ['gutenberg', 'kids.frontiersin', 'commonlit', 'simple.wikipedia', 'wikipedia', 'africanstorybook']
Pub Year (float64, 168 distinct): ['2020.0', '2019.0', '2017.0', '1915.0', '1881.0', '2018.0', '1883.0', '1882.0', '1922.0', '1914.0']
Category (object, 2 distinct): ['Lit', 'Info']
Location (object, 4 distinct): ['mid', 'start', 'whole', 'end']
License (object, 19 distinct): ['PD', 'CC BY 4.0', 'CC BY-SA 3.0', 'CC BY-SA 3.0 and GFDL', 'CC BY-NC-SA 2.0']
Excerpt (object, 4723 distinct): [...]
BT Easiness (float64, 4724 distinct): ['-0.3403', '-2.2739', '-3.4311', '-0.7553', '-0.8033']
BT s.e. (float64, 4724 distinct): ['0.464', '0.5147', '0.6002', '0.4847', '0.4646', '0.4792', '0.5265', '0.5072', '0.4814', '0.4626']
Flesch-Reading-Ease (float64, 3282 distinct): ['73.2', '61.33', '55.88', '75.69', '64.6', '75.43', '61.11', '57.61', '71.72', '58.03']
Flesch-Kincaid-Grade-Level (float64, 1593 distinct): ['10.14', '7.5', '11.81', '9.26', '11.52', '8.08', '10.37', '10.96', '11.39', '12.4']
Automated Readability Index (float64, 1789 distinct): ['10.43', '12.91', '13.79', '11.77', '11.32', '12.47', '11.38', '9.81', '12.14', '13.07']
SMOG Readability (float64, 29 distinct): ['11.0', '10.0', '12.0', '9.0', '8.0', '7.0', '13.0', '14.0', '6.0', '5.0']
New Dale-Chall Readability Formula (float64, 816 distinct): ['7.39', '7.89', '7.26', '7.36', '7.04', '6.98', '7.81', '7.65', '6.14', '6.91']
CAREC (float64, 4453 distinct): ['0.1549', '0.1507', '0.2075', '0.1015', '0.1802', '0.1917', '0.1439', '0.1594', '0.1725', '0.0977']
CAREC_M (float64, 4427 distinct): ['0.1617', '0.1795', '0.2069', '0.1177', '0.1444', '0.1749', '0.2708', '0.0549', '0.1386', '0.2272']
CARES (float64, 4718 distinct): ['0.5531', '0.4909', '0.6088', '0.4415', '0.4755', '0.4558', '0.3795', '0.3384', '0.4588', '0.5494']
CML2RI (float64, 4717 distinct): ['26.9921', '27.2612', '4.4395', '22.015', '16.7248', '8.6414', '8.3293', '10.8955', '14.7414', '5.2221']
Kaggle split (object, 2 distinct): ['Train', 'Test']
'''


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "CLEAR.csv")


CONTEXT = "Readability scores for text passages spanning various genres and time periods"
TARGET = CuratedTarget(raw_name="New Dale-Chall Readability Formula", task_type=SupervisedTask.REGRESSION)
COLS_TO_DROP = ['Kaggle split', 'firstPlace_pred', 'secondPlace_pred', 'thirdPlace_pred', 'fourthPlace_pred',
                'fifthPlace_pred', 'sixthPlace_pred', "ID", "Last Changed"]
FEATURES = []
LOADING_FUNC = load_df

DESCRIPTION = '''
About Dataset
📚 CommonLit Ease of Readability (CLEAR) Corpus 📚

Dive into this comprehensive dataset, curated by CommonLit in collaboration with Georgia State University. With approximately 5,000 reading passages spanning from the 3rd to 12th grade levels, this resource is a treasure trove for researchers, educators, and data enthusiasts alike.

🔍 Key Features:

Unique readability scores for each passage.
Text excerpts covering over 250 years of literature across various genres.
Meta-data including publishing year, genre, and more.
Readability indices such as Flesch-Reading-Ease, Flesch-Kincaid-Grade-Level, and predictions from the Kaggle Readability Prize.
An invaluable resource for NLP, education, and literary analysis.
🔄 Continuously updated by CommonLit, ensuring fresh and accurate data.

Whether you're developing a new readability metric, analyzing literary trends, or training a machine learning model, the CLEAR Corpus is your go-to dataset! 🚀
'''
