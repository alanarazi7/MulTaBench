from multabench.datasets.curation_objects import CuratedFeature, CuratedTarget
from multabench.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: jigsaw_unintended_bias100K
====
Examples: 100000
====
URL: https://www.openml.org/search?type=data&id=46654
====
Description: Predict whether online social media comments are toxic based on their text
    and additional tabular features providing information about the post (e.g. likes, rating, date created, etc.). This dataset originates from a 2019 Kaggle competition
    (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
    in which the 1st place solution utilized dataset-specific tricks such as a Bucket Sequencing Collator,
    auxiliary domain-specific prediction tasks for models, and a custom mimic loss function for training.

 Dataset found from the paper: Benchmarking multimodal automl for tabular data with text fields. arXiv preprint arXiv:2111.02705.
====
Target Variable: target (nominal, 2 distinct): ['0', '1']
====
Features:

comment_text (object, 99661 distinct, 0.0% missing): ['No.', 'Well said!', 'Well said.', 'Thank you.', '.', 'Sᴛᴀʀᴛ ᴡᴏʀᴋɪɴɢ ғʀᴏᴍ ʜᴏᴍᴇ! Gʀᴇᴀᴛ ᴊᴏʙ ғᴏʀ sᴛᴜᴅᴇɴᴛs, sᴛᴀʏ-ᴀᴛ-ʜᴏᴍᴇ ᴍᴏᴍs ᴏʀ ᴀɴʏᴏɴᴇ ɴᴇᴇᴅɪɴɢ ᴀɴ ᴇxᴛʀᴀ ɪɴᴄᴏᴍᴇ... Yᴏᴜ ᴏɴʟʏ ɴᴇᴇᴅ ᴀ ᴄᴏᴍᴘᴜᴛᴇʀ ᴀɴᴅ ᴀ ʀᴇʟɪᴀʙʟᴇ ɪɴᴛᴇʀɴᴇᴛ ᴄᴏɴɴᴇᴄᴛɪᴏɴ... Mᴀᴋᴇ $90 ʜᴏᴜʀʟʏ ᴀɴᴅ ᴜᴘ ᴛᴏ $12000 ᴀ ᴍᴏɴᴛʜ ʙʏ ғᴏʟʟᴏᴡɪɴɢ ʟɪɴᴋ ᴀᴛ ᴛʜᴇ ʙᴏᴛᴛᴏᴍ ᴀɴᴅ sɪɢɴɪɴɢ ᴜᴘ... Yᴏᴜ ᴄᴀɴ ʜᴀᴠᴇ ʏᴏᴜʀ ғɪʀsᴛ ᴄʜᴇᴄᴋ ʙʏ ᴛʜᴇ ᴇɴᴅ ᴏғ ᴛʜɪs ᴡᴇᴇᴋ... \n\n+++++++++http://www.cashapp24.com/', 'Thank you!', 'What?', 'I agree.', 'Why?']
asian (float64, 26 distinct, 77.5% missing): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.5', '0.3', '0.4', '0.6', '0.7']
atheist (float64, 17 distinct, 77.5% missing): ['0.0', '1.0', '0.1', '0.75', '0.8', '0.8333', '0.1667', '0.6', '0.25', '0.7']
bisexual (float64, 17 distinct, 77.5% missing): ['0.0', '0.1', '0.2', '0.1667', '0.3', '0.4', '0.5', '0.3333', '1.0', '0.25']
black (float64, 23 distinct, 77.5% missing): ['0.0', '1.0', '0.8', '0.8333', '0.1', '0.6', '0.5', '0.7', '0.1667', '0.2']
buddhist (float64, 16 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.5', '1.0', '0.8333', '0.75', '0.6', '0.2', '0.25']
christian (float64, 27 distinct, 77.5% missing): ['0.0', '1.0', '0.4', '0.6', '0.3', '0.5', '0.8', '0.1667', '0.2', '0.8333']
female (float64, 33 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.1667', '0.8', '0.2', '0.1', '0.7', '0.3', '0.4']
heterosexual (float64, 20 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '1.0', '0.8333', '0.5', '0.6', '0.25', '0.8', '0.75']
hindu (float64, 17 distinct, 77.5% missing): ['0.0', '0.1667', '0.1', '1.0', '0.2', '0.8', '0.5', '0.8333', '0.25', '0.3']
homosexual_gay_or_lesbian (float64, 23 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.8', '0.1', '0.6', '0.7', '0.2', '0.1667', '0.9']
intellectual_or_learning_disability (float64, 11 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.25', '0.6', '0.5', '0.0008']
jewish (float64, 21 distinct, 77.5% missing): ['0.0', '1.0', '0.1', '0.8', '0.8333', '0.1667', '0.7', '0.2', '0.9', '0.5']
latino (float64, 22 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.4', '0.2', '0.25', '0.3', '1.0', '0.5', '0.7']
male (float64, 36 distinct, 77.5% missing): ['0.0', '1.0', '0.1667', '0.2', '0.1', '0.8333', '0.5', '0.8', '0.7', '0.6']
muslim (float64, 25 distinct, 77.5% missing): ['0.0', '1.0', '0.8333', '0.8', '0.5', '0.1', '0.6', '0.2', '0.4', '0.1667']
other_disability (float64, 12 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.1429', '0.3', '0.0016', '0.0008', '0.0006', '0.4']
other_gender (float64, 9 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.0016', '0.0011', '0.0006', '0.6']
other_race_or_ethnicity (float64, 28 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.4', '0.5', '0.3333', '0.6']
other_religion (float64, 20 distinct, 77.5% missing): ['0.0', '0.1', '0.2', '0.1667', '0.25', '0.3', '0.5', '0.4', '0.3333', '0.75']
other_sexual_orientation (float64, 14 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.25', '0.3', '0.1429', '0.3333', '0.0064', '0.0197']
physical_disability (float64, 14 distinct, 77.5% missing): ['0.0', '0.1', '0.1667', '0.2', '0.3', '0.4', '0.5', '0.25', '0.0008', '0.1429']
psychiatric_or_mental_illness (float64, 22 distinct, 77.5% missing): ['0.0', '0.1667', '1.0', '0.1', '0.2', '0.3', '0.4', '0.6', '0.5', '0.8333']
transgender (float64, 21 distinct, 77.5% missing): ['0.0', '0.1', '1.0', '0.2', '0.1667', '0.5', '0.3', '0.25', '0.8333', '0.6']
white (float64, 27 distinct, 77.5% missing): ['0.0', '1.0', '0.8', '0.8333', '0.7', '0.6', '0.1', '0.5', '0.1667', '0.9']
created_date (datetime64[ns], 99996 distinct, 0.0% missing): ['2015-09-29 17:37:25.068440', '2015-10-13 17:16:38.081524', '2015-10-06 18:23:45.995878', '2017-09-24 20:04:11.379178', '2016-10-31 16:50:29.711104', '2017-06-22 08:23:15.005369', '2017-04-04 19:01:00.146140', '2017-06-16 15:28:55.285999', '2017-07-25 01:15:39.369564', '2017-01-09 01:29:42.238578']
funny (uint8, 32 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
wow (uint8, 8 distinct): ['0', '1', '2', '3', '4', '5', '7', '15']
sad (uint8, 16 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
likes (uint8, 95 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
disagree (uint8, 49 distinct): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
'''

CONTEXT = "Online Social Media Comments Toxicity"
TARGET = CuratedTarget(raw_name="target", new_name="Is Toxic", task_type=SupervisedTask.BINARY)
COLS_TO_DROP = ["id", "severe_toxicity",
                "toxicity_annotator_count", "identity_annotator_count",
                "publication_id", "parent_id", "article_id", "rating",
                # https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
                # The following should be excluded: "The data also has several additional toxicity subtype attributes.
                # Models do not need to predict these attributes for the competition, they are included as an
                # additional avenue for research. Subtype attributes are:"
                "insult", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"
                ]
FEATURES = [CuratedFeature(raw_name="created_date", feat_type=FeatureType.DATE)]
