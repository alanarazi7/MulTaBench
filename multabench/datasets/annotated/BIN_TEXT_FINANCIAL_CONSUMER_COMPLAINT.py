from os.path import join

from pandas import DataFrame, read_csv
from tabstar.constants import SEED

from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: selener/consumer-complaint-database/
====
Examples: 1282348
====
URL: https://www.kaggle.com/selener/consumer-complaint-database
====
Description:
Consumer Complaints Dataset This dataset contains records of consumer complaints filed against financial service
companies, featuring structured fields such as product type, state, and submission method, alongside rich textual
attributes like sub-product, issue, sub-issue, and company name. The prediction target is the company's response to the
complaint, consolidated into four meaningful classes: CLOSED WITH EXPLANATION, CLOSED WITH NON-MONETARY
RELIEF, CLOSED WITH MONETARY RELIEF, AND CLOSED WITHOUT RELIEF. This focused setup maintains a
challenging multi-class classification task while allowing for a fair assessment of how textual embeddings can enhance
tabular model performance in real-world customer service scenarios. The original dataset has approx. 1.42m rows,
which we randomly downsample to 100k to include a fair representation of rare classes while keeping the size feasible
for training.


About Dataset
Context
These are real world complaints received about financial products and services. Each complaint has been labeled with a specific product; therefore, this is a supervised text classification problem. With the aim to classify future complaints based on its content, we used different machine learning algorithms can make more accurate predictions (i.e., classify the complaint in one of the product categories)

Content
The dataset contains different information of complaints that customers have made about a multiple products and services in the financial sector, such us Credit Reports, Student Loans, Money Transfer, etc.
The date of each complaint ranges from November 2011 to May 2019.

Acknowledgements
This work is considered a U.S. Government Work. The dataset is public dataset and it was downloaded from
https://catalog.data.gov/dataset/consumer-complaint-database
on 2019, May 13.

Inspiration
This is a sort of tutorial for beginner

====
Target Variable: Company response to consumer (object, 8 distinct): ['Closed with explanation', 'Closed with non-monetary relief', 'Closed with monetary relief', 'Closed without relief', 'Closed', 'In progress', 'Untimely response', 'Closed with relief']
====
Features:

Date received (datetime64[ns], 0 distinct): ['2017-09-08 00:00:00', '2017-09-09 00:00:00', '2017-01-19 00:00:00', '2017-01-20 00:00:00', '2017-09-13 00:00:00', '2018-04-05 00:00:00', '2017-09-12 00:00:00', '2018-04-10 00:00:00', '2017-09-11 00:00:00', '2017-09-14 00:00:00']
Product (object, 18 distinct): ['Mortgage', 'Debt collection', 'Credit reporting, credit repair services, or other personal consumer reports', 'Credit reporting', 'Credit card', 'Bank account or service', 'Student loan', 'Credit card or prepaid card', 'Checking or savings account', 'Consumer Loan']
Sub-product (object, 76 distinct): ['Credit reporting', 'Checking account', 'Other mortgage', 'Conventional fixed mortgage', 'I do not know', 'Other (i.e. phone, health club, etc.)', 'General-purpose credit card or charge card', 'FHA mortgage', 'Other debt', 'Conventional home mortgage']
Issue (object, 167 distinct): ['Incorrect information on your report', 'Loan modification,collection,foreclosure', 'Incorrect information on credit report', 'Loan servicing, payments, escrow account', "Cont'd attempts collect debt not owed", "Problem with a credit reporting company's investigation into an existing problem", 'Attempts to collect debt not owed', 'Account opening, closing, or management', 'Communication tactics', 'Improper use of your report']
Sub-issue (object, 218 distinct): ['Information belongs to someone else', 'Account status', 'Their investigation did not fix an error on your report', 'Debt is not mine', 'Information is not mine', 'Account status incorrect', 'Debt was paid', 'Account information incorrect', 'Debt is not yours', 'Not given enough info to verify debt']
Consumer complaint narrative (object, 366941 distinct): ['There are many mistakes appear in my report without my understanding.', ...]
Company public response (object, 10 distinct): ['Company has responded to the consumer and the CFPB and chooses not to provide a public response', ...]
Company (object, 5275 distinct): ['EQUIFAX, INC.', 'Experian Information Solutions Inc.', 'TRANSUNION INTERMEDIATE HOLDINGS, INC.', ...]
State (object, 63 distinct): ['CA', 'FL', 'TX', 'NY', 'GA', 'IL', 'NJ', 'PA', 'NC', 'OH']
ZIP code (object, 22591 distinct): ['300XX', '770XX', '330XX', '331XX', '606XX', '750XX', '334XX', '303XX', '945XX', '900XX']
Tags (object, 3 distinct): ['Servicemember', 'Older American', 'Older American, Servicemember']
Consumer consent provided? (object, 4 distinct): ['Consent provided', 'Consent not provided', 'Other', 'Consent withdrawn']
Submitted via (object, 6 distinct): ['Web', 'Referral', 'Phone', 'Postal mail', 'Fax', 'Email']
Date sent to company (datetime64[ns], 0 distinct): ['2017-09-08 00:00:00', '2017-09-09 00:00:00', '2017-01-19 00:00:00', '2017-09-13 00:00:00', '2017-01-20 00:00:00', '2017-09-14 00:00:00', '2017-01-24 00:00:00', '2018-04-10 00:00:00', '2019-04-02 00:00:00', '2019-04-16 00:00:00']
Timely response? (object, 2 distinct): ['Yes', 'No']
Consumer disputed? (object, 2 distinct): ['No', 'Yes']
'''

# The raw complaint database has 1.28M rows; sampled at load time so the upload matches.
MAX_ROWS = 100_000


def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "rows.csv")
    df = read_csv(df_path)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return df



CONTEXT = "Real world complaints of consumer over financial issues"
TARGET = CuratedTarget(raw_name="Company response to consumer", task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name="Date received", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Date sent to company", feat_type=FeatureType.DATE),]
COLS_TO_DROP = ['Complaint ID']
IMAGE_FOLDER = None
LOADING_FUNC = load_df
PROCESSING_FUNC = None
