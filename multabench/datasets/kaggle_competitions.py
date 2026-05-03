import os

from multabench.constants import KAGGLE_KEY, KAGGLE_USERNAME


KAGGLE_CACHE_DIR = ".tabstar_kaggle_cache"

def login_to_kaggle():
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api



def download_kaggle_competition(competition: str):
    print(f"🏁 Downloading Kaggle competition dataset {competition}")
    api = login_to_kaggle()
    from kaggle.api.kaggle_api_extended import KaggleApi
    assert isinstance(api, KaggleApi)
    api.competition_download_files(competition, path=KAGGLE_CACHE_DIR, quiet=False)