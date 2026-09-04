"""Per-dataset target binning for the regression-heavy MulTaBench-Full text additions.

The text half of MulTaBench-Full came out 16-of-20 regression, so each regression target is also
evaluated as classification with its target cut into equal-frequency quantile bins. Each dataset
gets ONE bin count, sampled once from {2, 3, 5, 10} with a balanced random assignment (each count
used 4 times, seed 20260901) and frozen here so runs are reproducible.

Used by `benchmark.py --target_bins auto`.
"""

TARGET_BINS_BY_DATASET = {
    "REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES": 10,
    "REG_TEXT_SOCIAL_KOREAN_DRAMA": 10,
    "REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA": 5,
    "REG_TEXT_FOOD_WINE_VIVINO_SPAIN": 5,
    "REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS": 2,
    "REG_TEXT_SOCIAL_ANIME_PLANET_RATING": 2,
    "REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN": 5,
    "REG_TEXT_SOCIAL_BOOKS_GOODREADS": 3,
    "REG_TEXT_FOOD_RAMEN_RATINGS_2022": 10,
    "REG_TEXT_SPORTS_FIFA22_WAGES": 10,
    "REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020": 2,
    "REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES": 3,
    "REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES": 5,
    "REG_TEXT_HOUSES_AIRBNB_SEATTLE": 2,
    "REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE": 3,
    "REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES": 3,
}


def bins_for_dataset(dataset_name: str) -> int:
    n_bins = TARGET_BINS_BY_DATASET.get(dataset_name)
    assert n_bins is not None, (
        f"No bin count assigned for {dataset_name}. Add one to TARGET_BINS_BY_DATASET in "
        f"multabench/datasets/target_bins.py, or pass --target_bins <int> explicitly."
    )
    return n_bins
