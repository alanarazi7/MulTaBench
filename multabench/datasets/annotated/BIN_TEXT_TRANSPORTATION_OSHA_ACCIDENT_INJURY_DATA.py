from os.path import join

from pandas import DataFrame, read_csv

from multabench.datasets.curation_objects import CuratedTarget, CuratedFeature
from multabench.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: ruqaiyaship/osha-accident-and-injury-data-1517/
====
Examples: 4847
====
URL: https://www.kaggle.com/ruqaiyaship/osha-accident-and-injury-data-1517
====
Description:
Context
Health, Safety, and Environment (HSE) is a dicspline centered on implementing practices for environmental protection and safety in a workplace. Energy companies place a strong emphasis on HSE when conducting day to day operations, whether it is on the field or in an office. A major challenge with HSE, however, is monitoring and managing HSE incidents across an enterprise. The common practice for incident management is analyzing detailed incident reports. This can be cumbersome and time-consuming, because in most cases, these reports contain unstructured text. To increase efficiency, companies are seeking technologies that allow them to derive valuable insights from unstructured HSE data efficiently.

Content
This dataset contains abstracts of the accidents and injuries of construction workers from 2015-2017. There is some structured data around the unstructured text abstracts, such as Degree of Injury, Body Part(s) Affected, and Construction End Use.

Acknowledgements
This is OSHA data which is publicly available.

Inspiration
What are the most buildings/structures to build? What trends do we see in injuries in terms of time of day, time of year, etc.? What is the reason injuries are occurring? Where do we need more training and safety measures in place?
====
Target Variable: Task Assigned (object, 2 distinct): ['Regularly Assigned', 'Not Regularly Assigned']
====
Features:

Event Date (datetime64[ns], 0 distinct): ['2017-04-10 00:00:00', '2017-01-17 00:00:00', '2017-02-10 00:00:00', '2017-03-16 00:00:00', '2017-01-18 00:00:00', '2017-04-13 00:00:00', '2017-03-15 00:00:00', '2017-02-01 00:00:00', '2017-01-10 00:00:00', '2017-04-11 00:00:00']
Abstract Text (object, 4829 distinct): ['At 1:45 p.m. on January 9, 2017, an employee of a poultry processor was working in an engine room.  The employee observed a leak of anhydrous ammonia and took action.  No one was injured in this incident, which garnered considerable publicity. ', ...]
Event Description (object, 4320 distinct): ['EMPLOYEE FALLS FROM ROOF AND IS KILLED', 'EMPLOYEE FALLS FROM LADDER AND IS KILLED', 'EMPLOYEE DIES FROM HEART ATTACK', ...]
Event Keywords (object, 4427 distinct): ['HEART ATTACK', 'FALL,LADDER', 'FALL,ROOF', ...]
Construction End Use (object, 18 distinct): [' ', 'Commercial building', 'Single family or duplex dwelling', 'Other building', 'Multi-family dwelling', 'Highway, road, street', 'Manufacturing plant', 'Pipeline', 'Other heavy construction', 'Sewer/water treatment plant']
Building Stories (object, 25 distinct): [' ', '1', '2', '3', '4', '5', '6', '7', '10', '8']
Project Cost (object, 8 distinct): [' ', 'Under $50,000', '$1,000,000 to $5,000,000', '$500,000 to $1,000,000', '$50,000 to $250,000', '$250,000 to $500,000', '$10,000,000 to $20,000,000', '$5,000,000 to $10,000,000']
Project Type (object, 6 distinct): [' ', 'New project or new addition', 'Alteration or rehabilitation', 'Maintenance or repair', 'Demolition', 'Other']
Degree of Injury (object, 2 distinct): ['Fatal', 'Nonfatal']
Nature of Injury (object, 19 distinct): ['Serious Fall/Strike', 'Fracture, Broken Bones', 'Amputation, Crushing', 'Laceration', 'Head Trauma', 'Bruising, Contusion', 'Asphyxiation, Drowning', 'Electrocution', 'Fire Burn', 'Dislocation']
Part of Body (object, 29 distinct): ['Head', 'Whole Body', 'Fingers', 'Internal Injuries', 'Heart', 'Ribs', 'Left Leg', 'Hand', 'Neck', 'Feet']
Event type (object, 14 distinct): ['Fall (from elevation)', 'Struck-by', 'Caught in or between', 'Other', 'Shock', 'Card-vascular/resp. fail.', 'Struck against', 'Fall (same level)', 'Inhalation', 'Bite/sting/scratch']
Environmental Factor (object, 17 distinct): ['Other', 'Materials Handling Equip./Method', 'Work-Surface/Facility-Layout Condition', 'Pinch Point Action', 'Overhead Moving/Falling Object Action', 'Catch Point/Puncture Action', 'Flying Object Action', 'Shear Point Action', 'Weather, Earthquake, Etc.', 'Gas/Vapor/Mist/Fume/Smoke/Dust']
Human Factor (object, 18 distinct): ['Other', 'Misjudgment, Hazardous Situation', 'Safety Devices Removed/Inoperable', 'Position Inappropriate For Task', 'Mater-Handling Procedure Inappropriate', 'Insufficient /Lack/Engineering Controls', 'Insufficient /Lack/Protective Work Clothing/Equipment', 'Insufficient /Lack/Written Work Practice Program', 'Equipment Inappropriate For Operation', 'Lockout/Tagout Procedure Malfunction']
hazsub (object, 31 distinct): ['0', '8870', '1900', '8330', '8880', '170', '640', '614', 'D150', '240']
fat_cause (int64, 30 distinct): ['0', '15', '26', '16', '14', '18', '20', '30', '25', '7']
'''

def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, 'OSHA HSE DATA_ALL ABSTRACTS 15-17_FINAL.csv')
    df = read_csv(df_path)
    return df


CONTEXT = "Occupational Safety and Health Administration (OSHA) injury prediction during employment"
TARGET = CuratedTarget(raw_name="Task Assigned", task_type=SupervisedTask.BINARY)
FEATURES = [CuratedFeature(raw_name='Event Date', feat_type=FeatureType.DATE),]
COLS_TO_DROP = [
    # Columns which are duplicate, once having a semantic meaning and once just an index
    'task_assigned', 'part_of_body', 'nature_of_inj', 'proj_type', 'proj_cost', 'con_end', 'event_type',
    'evn_factor', 'hum_factor', 'build_stor',
    # Just an ID
    'summary_nr',
    # Constant value
    'fall_ht',
]
IMAGE_FOLDER = None
LOADING_FUNC = load_df
PROCESSING_FUNC = None
