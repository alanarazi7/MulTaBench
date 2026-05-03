from pandas import DataFrame, Series

from tabstar.constants import SEED
from tabstar.tabstar_datasets import TEXT2FOLD
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

from multabench.baselines.abstract_model import TabularModel


NEW2PAPER = {
    'BIN_IMAGE_CELEB_ATTRACTIVENESS': None,
    'BIN_IMAGE_HATEFUL_MEME': None,
    'BIN_IMAGE_MAMMOGRAPHY_CMMD': None,
    'MUL_IMAGE_CBIS_DDSM': None,
    'MUL_IMAGE_CHEXPERT': None,
    'MUL_IMAGE_CSGO_SKIN_PRICE': None,
    'MUL_IMAGE_FLOWER_BOUQUETS': None,
    'MUL_IMAGE_GLAUCOMA_SMDG': None,
    'MUL_IMAGE_HUBMAP_HPA': None,
    'MUL_IMAGE_JUSTIN_INSTAGRAM': None,
    'MUL_IMAGE_PETFINDER': None,
    'MUL_IMAGE_ZOOSCAN_ZOOPLANKTON': None,
    'REG_IMAGE_AMAZON_BEST_SELLER': None,
    'REG_IMAGE_AMAZON_PACKAGES': None,
    'REG_IMAGE_HNM_FASHION': None,
    'REG_IMAGE_KHAADI_CLOTHES': None,
    'REG_IMAGE_LETTERBOXD_MOVIES': None,
    'REG_IMAGE_MANGO_MASS': None,
    'REG_IMAGE_MKPHOTO_BOTS': None,
    'REG_IMAGE_PAINTING_PRICE': None,
    'BIN_TEXT_FAKE_JOB_POSTING': 'BIN_PROFESSIONAL_FAKE_JOB_POSTING',
    'BIN_TEXT_JIGSAW_TOXICITY': 'BIN_SOCIAL_JIGSAW_TOXICITY',
    'BIN_TEXT_KICKSTARTER_FUNDING': 'BIN_PROFESSIONAL_KICKSTARTER_FUNDING',
    'MUL_TEXT_DATA_SCIENTIST_SALARY': 'MUL_PROFESSIONAL_DATA_SCIENTIST_SALARY',
    'MUL_TEXT_MICHELIN_RESTAURANTS': 'MUL_FOOD_MICHELIN_GUIDE_RESTAURANTS',
    'MUL_TEXT_PRODUCT_SENTIMENT': 'MUL_CONSUMER_PRODUCT_SENTIMENT',
    'MUL_TEXT_SPOTIFY_GENRES': 'REG_SOCIAL_SPOTIFY_POPULARITY',
    'MUL_TEXT_US_ACCIDENTS': 'MUL_TRANSPORTATION_US_ACCIDENTS_MARCH23',
    'MUL_TEXT_WINE_REVIEW': 'MUL_FOOD_WINE_REVIEW',
    'MUL_TEXT_WOMEN_CLOTHING_REVIEW': 'MUL_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW',
    'REG_TEXT_BABIES_PRICES': 'REG_CONSUMER_BABIES_R_US_PRICES',
    'REG_TEXT_BOOK_PRICE': 'REG_CONSUMER_BOOK_PRICE_PREDICTION',
    'REG_TEXT_BOOK_READABILITY': 'REG_SOCIAL_BOOK_READABILITY_CLEAR',
    'REG_TEXT_MERCARI_MARKETPLACE': 'REG_CONSUMER_MERCARI_ONLINE_MARKETPLACE',
    'REG_TEXT_MONTGOMERY_SALARIES': 'REG_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY',
    'REG_TEXT_SCIMAGOJR_IMPACT': 'REG_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT',
    'REG_TEXT_ROTTEN_TOMATOES': 'REG_SOCIAL_MOVIES_ROTTEN_TOMATOES',
    'REG_TEXT_VANCOUVER_SALARIES': 'REG_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER',
    'REG_TEXT_VIDEO_GAMES_SALES': 'REG_SOCIAL_VIDEO_GAMES_SALES',
    'REG_TEXT_ZOMATO_RESTAURANTS': 'REG_FOOD_ZOMATO_RESTAURANTS'
}


def get_tabstar_paper_mapping(dataset_id: str) -> str | None:
    assert dataset_id in NEW2PAPER
    dataset = NEW2PAPER[dataset_id]
    if dataset:
        assert dataset in TEXT2FOLD, f"{dataset} not in {TEXT2FOLD}"
    return dataset



class TabSTAR(TabularModel):

    MODEL_NAME = "TabSTAR ⭐"
    SHORT_NAME = "tabstar"
    USE_VAL_SPLIT = False
    USE_MEDIAN_FILLING = False
    USE_CATEGORICAL_ENCODING = False
    USE_TEXT_EMBEDDINGS = False
    USE_TARGET_ENCODER = False

    IS_PAPER_VERSION = True

    def initialize_model(self) -> TabSTARClassifier | TabSTARRegressor:
        # TODO: if memory is low, use smaller batch size?
        lora_batch = 64
        tabstar_cls = TabSTARClassifier if self.is_cls else TabSTARRegressor
        dataset_mapping = get_tabstar_paper_mapping(dataset_id=self.dataset.name)
        model = tabstar_cls(pretrain_dataset_or_path=dataset_mapping,
                            device=self.device, verbose=True, random_state=SEED,
                            is_paper_version=False,
                            lora_batch=lora_batch)
        return model


    def fit_model(self, x_train: DataFrame, y_train: Series, x_val: DataFrame, y_val: Series):
        # TODO: consider wrapping with retries due to CUDA errors
        self.model_.fit(x_train, y_train)
