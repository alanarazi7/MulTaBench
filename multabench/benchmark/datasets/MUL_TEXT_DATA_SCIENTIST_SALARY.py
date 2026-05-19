"""
Dataset Name: MUL_TEXT_DATA_SCIENTIST_SALARY
====
Examples: 15841
====
URL: https://www.openml.org/search?type=data&id=46664
====
Target Variable: salary (object, 6 distinct): ['10-15', '15-25', '6-10', '0-3', '3-6', '25-50']
====
Features:

experience (object, 128 distinct): ['5-10 yrs', '2-5 yrs', '3-8 yrs', '2-7 yrs', '3-5 yrs', '4-9 yrs', '3-6 yrs', '7-12 yrs', '1-3 yrs', '5-8 yrs']
job_description (object, 7859 distinct, 22.1% missing): ['Accenture Technology powers our clients businesses with innovative technologies established and emerging ...', '- Experience in Credit card/ banking domain with knowledge across customer lifecycle is must;- Candidate ...', '- Experience in defining and executing professional software engineering best practices for the full ...', '- An advanced degree in Math, Computer Science, Statistics, Physics, or a related field (high GPAs ...', '- Team management / mentor ship experience is must; Should be good at resolving conflicts;- Experience ...', '- Good team management, project management and communication (both written and verbal) skills, including ...', '- Post-Graduate degree in statistics, finance, mathematics, engineering (Computer Science preferred) or ...', 'Utilize strong analytical ability to evaluate end-to-end customer experience across multiple channels ...', 'Experience leading teams of size 5-15 members;Very good knowledge of statistical techniques such as ...', '- Experience in banking domain with knowledge across customer lifecycle is must;- Candidate should have ...']
job_desig (object, 10097 distinct): ['Business Analyst', 'Data Scientist', 'Data Analyst', 'Home Base Job/ Data Entry/online Work/part Time Work/freelancer work', 'Digital Marketing Manager', 'Product Manager', 'Digital Marketing Executive', 'Analyst', 'SEO Executive', 'SEO Analyst']
job_type (object, 5 distinct, 75.8% missing): ['Analytics', 'analytics', 'ANALYTICS', 'analytic', 'Analytic']
key_skills (object, 11155 distinct, 0.0% missing): ['part time, freelancing, data entry, present job, work from home...', 'SAS, Sdtm, Adam, Statistical Programming, Statistics, Life Sciences...', 'Ar Calling, ar analyst, accounts receivable, revenue cycle management...', 'Fraud Analytics, People Management Skills, Team Leading, Problem Solving...', 'SAS, Logistic Regression, Chaid, R, Data Analytics, Anova, Excel...', 'Communication Skills, Analytical, Problem Solving, itil solving...', 'Excel, SQL, Data Analysis, Segmentation, SAS, Data Mining, SPSS...', 'Analytics, SAS, banking, insurance, Analytics Head', 'data entry operation, typing, excel, notepad, freelancing, content writing,...', 'Linear Regression, Insurance Analytics, Business Analysis...']
location (object, 1355 distinct): ['Bengaluru', 'Mumbai', 'Gurgaon', 'Pune', 'Hyderabad', 'Chennai', 'Delhi NCR', 'Noida', 'Delhi NCR, Gurgaon', 'Delhi']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "MUL_TEXT_DATA_SCIENTIST_SALARY"
SLUG_BASE = "multabench-data-scientist-salary"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=46664"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.MUL_TEXT_PROFESSIONAL_DATA_SCIENTIST_SALARY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
