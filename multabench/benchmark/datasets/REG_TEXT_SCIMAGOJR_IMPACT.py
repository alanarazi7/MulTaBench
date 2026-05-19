"""
Dataset Name: REG_TEXT_SCIMAGOJR_IMPACT
====
Examples: 31136
====
Target Variable: H index (int64, 451 distinct): ['5', '4', '6', '3', '7', '8', '9', '2', '11', '10']
====
Features:

Rank (int64, 31136 distinct): ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
Title (object, 31120 distinct): ['Cahiers de Linguistique Asie Orientale', 'Principia', 'Neonatology', 'Engineering Journal', 'Nursing Management', 'Portal', 'Agenda', 'Philosophical Magazine', 'Internet of Things', 'Environmental Chemistry']
Type (object, 4 distinct): ['journal', 'book series', 'conference and proceedings', 'trade journal']
Issn (object, 31125 distinct): ['-', '17527074, 17527066', '09767428, 09761799', '27884538, 27884546', '26613190', '26375192', '23504269, 18558453', '17456436, 17456444', '17402123, 17402131', '2500106X']
SJR (object, 2968 distinct, 1.0% missing): ['0,101', '0,100', '0,102', '0,133', '0,111', '0,103', '0,116', '0,104', '0,105', '0,110']
SJR Best Quartile (object, 5 distinct): ['Q1', 'Q2', 'Q3', 'Q4', '-']
Total Docs  (2024) (int64, 1273 distinct): ['0', '20', '16', '21', '24', '14', '19', '15', '17', '10']
Total Docs  (3years) (int64, 2218 distinct): ['60', '65', '49', '72', '48', '63', '52', '54', '67', '66']
Total Refs (int64, 10230 distinct): ['0', '549', '1051', '878', '920', '257', '853', '720', '857', '1473']
Total Citations (3years) (int64, 4485 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Citable Docs  (3years) (int64, 2136 distinct): ['0', '1', '60', '71', '40', '63', '59', '70', '47', '53']
Citations / Doc  (2years) (object, 1429 distinct): ['0,00', '0,13', '0,11', '0,06', '0,07', '0,05', '0,08', '0,04', '0,21', '0,19']
Ref  / Doc (object, 8583 distinct): ['0,00', '34,00', '37,00', '39,00', '40,00', '28,00', '32,00', '33,00', '43,00', '29,00']
%Female (object, 1 distinct): ['-']
Overton (int64, 66 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '9', '8']
SDG (int64, 700 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Country (object, 120 distinct): ['United States', 'United Kingdom', 'Netherlands', 'Germany', 'China', 'Switzerland', 'Spain', 'Italy', 'Russian Federation', 'Poland']
Region (object, 9 distinct): ['Western Europe', 'Northern America', 'Asiatic Region', 'Eastern Europe', 'Latin America', 'Middle East', 'Pacific Region', 'Africa', 'Africa/Middle East']
Publisher (object, 8340 distinct, 2.4% missing): ['Taylor and Francis Ltd.', 'Elsevier B.V.', 'Routledge', 'Elsevier Ltd', 'SAGE Publications Ltd', 'SAGE Publications Inc.', 'Brill Academic Publishers', 'Oxford University Press', 'Wiley-Blackwell Publishing Ltd', 'Cambridge University Press']
Coverage (object, 6011 distinct): ['2019-2024', '2019-2025', '2020-2024', '2018-2024', '2008-2025', '2010-2025', '1996-2025', '2017-2024', '2020-2025', '2011-2024']
Categories (object, 17157 distinct): ['Medicine (miscellaneous) (Q4)', 'Medicine (miscellaneous) (Q3)', 'Linguistics and Language (Q2)', 'Law (Q4)', 'Education (Q1)', 'Law (Q3)', 'Education (Q2)', 'Literature and Literary Theory (Q4)', 'Education (Q3)', 'History (Q4)']
Areas (object, 1296 distinct): ['Medicine', 'Social Sciences', 'Arts and Humanities; Social Sciences', 'Arts and Humanities', 'Agricultural and Biological Sciences', 'Engineering', 'Mathematics', 'Biochemistry, Genetics and Molecular Biology; Medicine', 'Earth and Planetary Sciences', 'Computer Science']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import UrlDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_SCIMAGOJR_IMPACT"
SLUG_BASE = "multabench-scimagojr-impact"
KAGGLE_SOURCE = "https://www.scimagojr.com/journalrank.php?out=xls"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(UrlDatasetID.REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
