from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedTarget
from multabench.datasets.objects import SupervisedTask
from multabench.datasets.utils import load_csv

'''
Dataset Name: hernan4444/animeplanet-recommendation-database-2020
====
Examples: 16621
====
URL: https://www.kaggle.com/hernan4444/animeplanet-recommendation-database-2020
====
Description:
Anime-Planet Recommendation Database 2020
Recommendation data from 74.000 users and 16.000 animes at Anime-Planet

This dataset contains information about 16.621 anime, 175.731 recommendations and the preference from 74.129 different users of animes scrapped from anime-planet. In particular, this dataset contain:

Information about the anime like Tags, synopsis, average score, etc.
List of animes recommended given another anime and the count of user that are agreed with the recommendation.
HTML with anime information to do data scrapping. These files contain information such as reviews, synopsis, information about the staff, anime statistics, genre, etc.
the anime list per user. Include dropped, watched, want to watch, currently watching, stalled and Won't watch.
ratings given by users to the animes that they has watched completely.

Warning: this dataset includes information about anime for adults (hentai).
Content
The anime data was scrapped between June 4th and June 25th.

====
Features:

Anime-PlanetID (int64, 16621 distinct): ['10', '4824', '4792', '4793', '4794', '4796', '4797', '4798', '4799', '48']
Name (object, 16619 distinct): ['The Prince of Tennis', 'Arashi no Yoru ni: Himitsu no Tomodachi']
Alternative Name (object, 7290 distinct): ['Unknown', 'The Legend and The Hero', 'Long Zhi Gu: Jingling Wangzuo', 'Tennis no Ouji-sama']
Rating Score (object, 3517 distinct): ['Unknown', '2.25', '2.686', '2.95', '3.016', '2.652', '3.405', '3.409', '3.027', '2.519']
Number Votes (object, 4026 distinct): ['Unknown', '12', '10', '11', '13', '14', '15', '18', '16', '17']
Tags (object, 10782 distinct): ['Unknown', 'Vocaloid', 'Shorts', 'Minna no Uta', 'Family Friendly, Minna no Uta', 'Abstract']
Content Warning (object, 207 distinct): ['Unknown', 'Violence', 'Nudity', 'Explicit Violence', 'Explicit Sex', 'Nudity, Sexual Content']
Type (object, 8 distinct): ['TV', 'Movie', 'OVA', 'Web', 'Music', 'DVD', 'Other', 'TV\n(104']
Episodes (object, 215 distinct): ['1', '12', '13', '2', '26', 'Unknown', '3', '4', '6', '52']
Finished (bool, 2 distinct): ['1', '0']
Duration (object, 151 distinct): ['Unknown', '4', '2', '5', '3', '1', '6', '24', '10', '15']
StartYear (object, 104 distinct): ['2017', '2018', '2016', '2014', '2015', '2019', '2013', '2012', '2020', '2011']
EndYear (object, 104 distinct): ['2017', '2016', '2018', '2015', '2014', '2019', '2013', '2012', '2020', '2011']
Season (object, 111 distinct): ['Unknown', 'Spring 2018', 'Fall 2016', 'Spring 2016', 'Winter 2021', 'Spring 2017', 'Fall 2018']
Studios (object, 1045 distinct): ['Unknown', 'Toei Animation', 'Sunrise', 'J.C.Staff', 'TMS Entertainment', 'MADHOUSE', 'Studio DEEN']
Synopsis (object, 9067 distinct): ['No synopsis yet - check back soon!']
Url (object, 16621 distinct): ['https://www.anime-planet.com/anime/the-prince-of-tennis']
'''


def load_df(dir_path: str) -> DataFrame:
    return load_csv(dir_path, "anime.csv")


CONTEXT = "Anime-Planet Recommendation Database 2020"
TARGET = CuratedTarget(raw_name='Rating Score', task_type=SupervisedTask.REGRESSION, numeric_missing="Unknown")
COLS_TO_DROP = ["Anime-PlanetID", "Url"]
FEATURES = []
LOADING_FUNC = load_df

DESCRIPTION = '''
Anime-Planet Recommendation Database 2020
Recommendation data from 74.000 users and 16.000 animes at Anime-Planet

This dataset contains information about 16.621 anime, 175.731 recommendations and the preference from 74.129 different users of animes scrapped from anime-planet. In particular, this dataset contain:

Information about the anime like Tags, synopsis, average score, etc.
List of animes recommended given another anime and the count of user that are agreed with the recommendation.
HTML with anime information to do data scrapping. These files contain information such as reviews, synopsis, information about the staff, anime statistics, genre, etc.
the anime list per user. Include dropped, watched, want to watch, currently watching, stalled and Won't watch.
ratings given by users to the animes that they has watched completely.

Warning: this dataset includes information about anime for adults (hentai).
Content
The anime data was scrapped between June 4th and June 25th.

The "html" folder contain 1 zip per anime (16.621 different anime). Each zip contains different HTML pages scrapped from Anime-planet. The scrapped pages are:
Main page
Reviews
Recommendations
Characters
Staff
I uploaded 2 files as example to don't increase the size of this dataset. All HTML files are in this link: https://drive.google.com/drive/folders/1xIxBRtJR2oTZhJVvjFoTo3qllBFn4aOV?usp=sharing

animelist.csv have the list of all animes register by the user with the respective score, watching status and numbers of episodes watched. This dataset contains 20 Million row, 16.745 different animes and 74.129 different users. The file have the following columns:
user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
score: score between 1 to 5 given by the user in scale of 0.5. 0 if the user didn't assign a score. (e.g. 3.5)
watching_status: state ID from this anime in the anime list of this user. (e.g. 2)
watched_episodes: numbers of episodes watched by the user. (e.g. 24)
watching_status.csv describe every possible status of the column: "watching_status" in animelist.csv.

rating_complete.csv is a subset of animelist.csv. This dataset only considers animes that the user has watched completely (watching_status==1) and gave it a score (score!=0). This dataset contains 8 Million ratings applied to 15.681 animes by 68.199 users. This file have the following columns:

user_id: non identifiable randomly generated user id.
anime_id: Anime-planet ID of the anime. (e.g. 1).
rating: rating that this user has assigned.
anime_recommendations.csv have the list of all animes recommended given one anime. This information was scrapped from "recommendation" tab (e.g. https://www.anime-planet.com/anime/the-saints-magic-power-is-omnipotent/recommendations ). The file have the following columns:
Anime: Anime Planet ID of the anime. (e.g. 1).
Recommendation: Anime Planet ID of the recommended anime. (e.g. 1).
Agree Votes: number of users that was agreed with the recommendation.

anime.csv contain general information of every anime (16.621 different anime) like Tags, type, studio, synopsis, etc. This file have the following columns:

Anime-PlanetID: Anime Planet ID of the anime. (e.g. 1).
Name: full name of the anime. (e.g. FLCL)
Alternative Name: another way to call the anime. (e.g. Furi Kuri)
Rating Score: average score of the anime given from all users in Anime Planet database. (e.g. 8.78)
Number Votes: number of users who give a score to the anime. (e.g. 1241)
Tags: comma separated list of tags for this anime. (e.g. Comedy, Mecha, Sci Fi, Outer Space, Original Work)
Content Warning: comma separated list of content warning tags. (e.g. Explicit Violence, Mature Themes, Nudity)
Type: TV, movie, OVA, etc. (e.g. TV).
Episodes: number of chapters. (e.g. 26)
Finished: True if the anime finished when I did the data scraping. False is the anime is on going in that moment.
Duration: duration of the anime in minutes (e.g 60)
StartYear: year when the anime start the transmission. (e.g. 2016)
EndYear: year when the anime finish the transmission. (e.g. 2017)
Season: season and year of release (e.g. Fall 2000)
Studios: comma separated list of studios (e.g. Sunrise)
Synopsis: synopsis of the anime

Url: url to the main page of anime in Anime Planet (e.g. https://www.anime-planet.com/anime/vandread)
Acknowledgements
Thanks to:

Anime Planet for providing anime data.
Inspiration
Improve Anime Recommendation Database 2020 with more data like tags, content warning, another synopsis, etc.

Experiment with different types of recommended. For instance, collaborative filtering or based on context like Tags, synopsis, etc.

Use this information to build a better anime recommended system.

Identifying which feature allows us to build the best anime recommended system.

Build a second dataset with anime list per user.
'''
