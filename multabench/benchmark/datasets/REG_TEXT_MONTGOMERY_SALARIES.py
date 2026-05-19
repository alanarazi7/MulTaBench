"""
Dataset Name: REG_TEXT_MONTGOMERY_SALARIES
====
Examples: 9228
====
URL: https://www.openml.org/search?type=data&id=42125
====
Target Variable: current_annual_salary (float64, 3403 distinct): ['92756.7', '89620.0', '102664.0', '69222.18', '73801.0', '100849.36', '93396.0', '45261.0', '97912.0', '50910.0']
====
Features:

full_name (object, 9222 distinct): ['Wong, Ka Y.', 'Cruz, Angela', 'Carter, Jerome', 'Smith, Beverly E.', 'Miller, Michael E.', 'Smith, Jason M.', 'Nguyen-Vu, Diane V.', 'Nibber, Savita K.', 'Niblock, David K.', 'Nice, Matthew L.']
gender (object, 2 distinct, 0.2% missing): ['M', 'F']
2016_gross_pay_received (float64, 8977 distinct, 1.1% missing): ['119244.9', '120825.94', '625.0', '119244.91', '101542.68', '528.0', '103397.41', '22071.77', '69697.66', '0.0']
2016_overtime_pay (float64, 6176 distinct, 31.6% missing): ['0.01', '57.87', '73.64', '0.0', '94.51', '64.17', '68.73', '66.41', '132.31', '36.21']
department (object, 37 distinct): ['POL', 'HHS', 'FRS', 'DOT', 'COR', 'DLC', 'DGS', 'LIB', 'DPS', 'SHF']
department_name (object, 37 distinct): ['Department of Police', 'Department of Health and Human Services', 'Fire and Rescue Services', 'Department of Transportation', 'Correction and Rehabilitation', 'Department of Liquor Control', 'Department of General Services', 'Department of Public Libraries', 'Department of Permitting Services', "Sheriff's Office"]
division (object, 694 distinct): ['School Health Services', 'Transit Silver Spring Ride On', 'Transit Gaithersburg Ride On', 'Highway Services', 'Child Welfare Services', 'FSB Traffic Division School Safety Section', 'Income Supports', 'PSB 3rd District Patrol', 'PSB 4th District Patrol', 'Transit Nicholson Ride On']
assignment_category (object, 2 distinct): ['Fulltime-Regular', 'Parttime-Regular']
employee_position_title (object, 385 distinct): ['Police Officer III', 'Firefighter/Rescuer III', 'Bus Operator', 'Manager III', 'Correctional Officer III (Corporal)', 'Master Firefighter/Rescuer', 'Office Services Coordinator', 'School Health Room Technician I', 'Community Health Nurse II', 'Crossing Guard']
underfilled_job_title (object, 84 distinct, 88.2% missing): ['Firefighter/Rescuer II', 'Police Officer II', 'Police Officer I', 'Firefighter/Rescuer I (Recruit)', 'Correctional Officer II (PFC)', 'Public Safety Communications Specialist I', 'Public Safety Communications Specialist II', 'Supply Technician II', 'Permitting and Code Enforcement Inspector II', 'Permitting Services Specialist II']
date_first_hired (object, 2264 distinct): ['2016-12-12', '2013-01-14', '2014-02-24', '2014-03-10', '2013-08-12', '2014-10-06', '2014-09-22', '2007-03-19', '2013-07-29', '2012-07-16']
"""

import os

import pandas as pd

from multabench.datasets.all_datasets import OpenMLDatasetID
from multabench.datasets.downloading import download_dataset
from multabench.benchmark.utils.curation import save_dataset, task_type_from_name


DATASET_ID = "REG_TEXT_MONTGOMERY_SALARIES"
SLUG_BASE = "multabench-montgomery-salaries"
KAGGLE_SOURCE = "https://www.openml.org/search?type=data&id=42125"



def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dataset = download_dataset(OpenMLDatasetID.REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY)
    df = pd.concat([dataset.x, dataset.y], axis=1)
    save_dataset(df=df, output_dir=output_dir, target_col=dataset.y.name, dataset_id=DATASET_ID,
                 slug=slug, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)
