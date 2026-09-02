from multabench.datasets.all_datasets import KaggleDatasetID, UrlDatasetID, OpenMLDatasetID, MulTaBenchDatasetID
from multabench.datasets.tiers import MULTABENCH_FULL_TEXT_EXTRA, PROMOTED_FROM

# The ACCEPTED/REJECTED lists below record the MulTaBench-Core curation decision over the
# 56-dataset candidate pool: Joint Signal AND Tabular Awareness, 3 of 5 curation models at
# delta=0.001. They are hand-maintained but reproducible -- REJECTED_TEXT_DATASETS is exactly the
# set of "reject" rows of leaderboard/results/analysis_curation_sensitivity/text_full_selection.csv
# (column core_decision). MulTaBench-Full uses the weaker Joint-Signal-only rule; see the bottom
# of this file and multabench/leaderboard/analysis/text_full_selection.py.


# Benchmarking Multimodal AutoML for Tabular Data with Text Fields
# https://arxiv.org/abs/2111.02705

AUTOML_MULTIMODAL_ACCEPTED = [
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING,
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING,
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY,
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT,
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY,
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE,
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE,
    # Accepted, but too NLPish
    OpenMLDatasetID.MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY,
]

AUTOML_MULTIMODAL_REJECTED = [
    OpenMLDatasetID.BIN_TEXT_SOCIAL_IMDB_GENRE_PREDICTION,
    OpenMLDatasetID.MUL_TEXT_HOUSES_MELBOURNE_AIRBNB,
    OpenMLDatasetID.MUL_TEXT_SOCIAL_GOOGLE_QA_TYPE_REASON,
    OpenMLDatasetID.REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES,
    OpenMLDatasetID.REG_TEXT_CONSUMER_JC_PENNEY_PRODUCT_PRICE,
    OpenMLDatasetID.REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020

]

AUTOML_MULTIMODAL = AUTOML_MULTIMODAL_ACCEPTED + AUTOML_MULTIMODAL_REJECTED

# Vectorizing string entries for data processing on tables: when are larger language models better?
# https://arxiv.org/pdf/2312.09634

VECTORIZING_ACCEPTED = [
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES,
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS,
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS,
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY,
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES,
]
VECTORIZING_REJECTED = [
    UrlDatasetID.REG_TEXT_CONSUMER_BIKE_PRICE_BIKEWALE,
    KaggleDatasetID.REG_TEXT_FOOD_RAMEN_RATINGS_2022,
    UrlDatasetID.REG_TEXT_SOCIAL_BOOKS_GOODREADS,
    KaggleDatasetID.REG_TEXT_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE,
]

VECTORIZING = VECTORIZING_ACCEPTED + VECTORIZING_REJECTED

# CARTE: Pretraining and Transfer for Tabular Learning
# https://arxiv.org/abs/2402.16785

CARTE_ACCEPTED = [
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS,
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS,
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES,
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS,
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT,
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY,
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES,
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES,
]
CARTE_REJECTED = [
    KaggleDatasetID.MUL_TEXT_FOOD_YELP_REVIEWS,
    UrlDatasetID.REG_TEXT_CONSUMER_BIKE_PRICE_BIKEWALE,
    KaggleDatasetID.REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO,
    KaggleDatasetID.REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES,
    KaggleDatasetID.REG_TEXT_FOOD_BEER_RATINGS,
    KaggleDatasetID.REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS,
    KaggleDatasetID.REG_TEXT_FOOD_COFFEE_REVIEW,
    KaggleDatasetID.REG_TEXT_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE,
    OpenMLDatasetID.REG_TEXT_SPORTS_FIFA22_WAGES,
    KaggleDatasetID.REG_TEXT_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT,
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY,
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN,
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA,
    KaggleDatasetID.REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES,
    KaggleDatasetID.REG_TEXT_FOOD_WINE_VIVINO_SPAIN,
    KaggleDatasetID.REG_TEXT_FOOD_RAMEN_RATINGS_2022,
    KaggleDatasetID.REG_TEXT_SOCIAL_ANIME_PLANET_RATING,
    KaggleDatasetID.REG_TEXT_SOCIAL_FILMTV_MOVIE_RATING_ITALY,
    KaggleDatasetID.REG_TEXT_SOCIAL_KOREAN_DRAMA,
    KaggleDatasetID.REG_TEXT_SOCIAL_MUSEUMS_US_REVENUES,
    KaggleDatasetID.REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE,
    UrlDatasetID.REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,
]


CARTE_BENCHMARK = CARTE_ACCEPTED + CARTE_REJECTED


# Towards Benchmarking Foundation Models for Tabular Data With Text
# https://arxiv.org/abs/2507.07829

TEXT_TAB_BENCH_ACCEPTED = [
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING,
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING,
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES,
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE,
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    # Accepted, but weren't included in the final benchmark
    KaggleDatasetID.MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT,
    KaggleDatasetID.BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT,
]

TEXT_TAB_BENCH_REJECTED = [
    KaggleDatasetID.BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA,
    KaggleDatasetID.REG_TEXT_HOUSES_AIRBNB_SEATTLE,
    KaggleDatasetID.REG_TEXT_FOOD_BEER_RATINGS,
    OpenMLDatasetID.REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020,
    KaggleDatasetID.REG_TEXT_CONSUMER_LAPTOP_INDIAN_PRICES,
    KaggleDatasetID.REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS,
]


TEXT_TAB_BENCH = TEXT_TAB_BENCH_ACCEPTED + TEXT_TAB_BENCH_REJECTED


REJECTED_TEXT_DATASETS = TEXT_TAB_BENCH_REJECTED + AUTOML_MULTIMODAL_REJECTED + VECTORIZING_REJECTED + CARTE_REJECTED
ACCEPTED_TEXT_DATASETS = TEXT_TAB_BENCH_ACCEPTED + AUTOML_MULTIMODAL_ACCEPTED + VECTORIZING_ACCEPTED + CARTE_ACCEPTED

assert len(set(ACCEPTED_TEXT_DATASETS)) == 23, (
    f"Expected 23 unique accepted text datasets (20 in benchmark + 3 accepted-but-excluded), "
    f"got {len(set(ACCEPTED_TEXT_DATASETS))}"
)
assert len(set(ACCEPTED_TEXT_DATASETS) | set(REJECTED_TEXT_DATASETS)) == 56, (
    f"Expected 56 unique text-tabular candidate datasets total, "
    f"got {len(set(ACCEPTED_TEXT_DATASETS) | set(REJECTED_TEXT_DATASETS))}"
)


# --- MulTaBench-Full (Joint Signal only) ---------------------------------------------------
# The 20 pool datasets promoted from rejected/excluded to MulTaBench-Full: they show joint signal
# (the joint frozen model beats both unimodal models) even though fine-tuning the encoder adds
# nothing on most of them, which is what kept them out of Core.
TEXT_FULL_EXTRA = MULTABENCH_FULL_TEXT_EXTRA

# Passed Joint Signal too, but not selected. Kept here as the documented reserve, in selection-rank
# order, for swaps or if a selected dataset later has to be dropped.
#
# IMDB Genre and Melbourne Airbnb left this list: both were curated, uploaded and re-verified on
# their own artifacts, where they scored better than in the pool (5/5 and 4/5 vs 4/5 and 3/5), and
# the Full text half is targeting a 10/10 classification-regression balance that they help reach.
# The two regression datasets they displaced come back here -- they were the only selected members
# below unanimous agreement, and neither had been uploaded.
TEXT_FULL_NEAR_MISS = [
    KaggleDatasetID.REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE,          # joint 4/5, 8/10 (displaced)
    KaggleDatasetID.REG_TEXT_CONSUMER_LAPTOP_INDIAN_PRICES,          # joint 4/5, 7/10
    UrlDatasetID.REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,       # joint 4/5, 7/10 (displaced)
    KaggleDatasetID.REG_TEXT_SOCIAL_MUSEUMS_US_REVENUES,             # joint 3/5, 5/10
    KaggleDatasetID.REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS,  # joint 3/5, 4/10
]

_TEXT_POOL = set(ACCEPTED_TEXT_DATASETS) | set(REJECTED_TEXT_DATASETS)


def _pool_source(dataset_id):
    """Once a Full member is re-hosted it carries a MulTaBenchDatasetID, which is not a pool id.
    Resolve it back to the source it was promoted from so the pool check keeps working."""
    return PROMOTED_FROM.get(dataset_id, dataset_id)


_selected_sources = {_pool_source(d) for d in TEXT_FULL_EXTRA}
assert _selected_sources <= _TEXT_POOL, (
    "Full-tier text datasets must come from the 56-dataset candidate pool: "
    f"{[d.name for d in _selected_sources - _TEXT_POOL]}"
)
assert set(TEXT_FULL_NEAR_MISS) <= _TEXT_POOL, (
    "Near-miss text datasets must come from the 56-dataset candidate pool: "
    f"{[d.name for d in set(TEXT_FULL_NEAR_MISS) - _TEXT_POOL]}"
)
assert not _selected_sources & set(TEXT_FULL_NEAR_MISS), (
    "A dataset cannot be both selected for Full and held in reserve: "
    f"{[d.name for d in _selected_sources & set(TEXT_FULL_NEAR_MISS)]}"
)
