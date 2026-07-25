# Committee panel pass rates — per-dataset summary

All 56 text-tabular pool datasets, at the paper's default curation setting (5-model committee, delta=0.001, rho=3/5). Derived from `committee_delta_sweep.csv`.

- **23** accept / **33** reject (paper's 3-of-5 rule at delta=0.001)
- **20** of the 23 accepted datasets are in the final MulTaBench benchmark (3 accepted-but-excluded to match the image subset size)

**Columns**

- `panel_pass_rate_ge3` — share of *all* possible eligible-model 5-panels (C(10,5)=252, or C(8,5)=56 for the 2 TabPFN-ineligible datasets) that would independently accept the dataset at a 3-of-5 quorum. Analyses (a) model committee and (c) quorum size.
- `flips_at_delta` — for currently-accepted datasets, the smallest delta in {0.002 … 0.05} at which the original committee's decision flips to reject. Blank where no flip occurs in that range (and for all already-rejected datasets). Analysis (b) effect-size threshold.

| Dataset | Decision | In benchmark | Panel pass rate (≥3/5) | Flips at δ ≥ |
|---|---|:---:|---:|---:|
| `BIN_TEXT_PROFESSIONAL_KICKSTARTER_FUNDING` | accept | ✓ | 100.0% | 0.02 |
| `BIN_TEXT_SOCIAL_JIGSAW_TOXICITY` | accept | ✓ | 100.0% | 0.01 |
| `MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT` | accept | ✓ | 100.0% | 0.02 |
| `MUL_TEXT_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW` | accept | ✓ | 100.0% | 0.01 |
| `MUL_TEXT_FOOD_MICHELIN_GUIDE_RESTAURANTS` | accept | ✓ | 100.0% | 0.02 |
| `MUL_TEXT_FOOD_WINE_REVIEW` | accept | ✓ | 100.0% | 0.002 |
| `MUL_TEXT_PROFESSIONAL_DATA_SCIENTIST_SALARY` | accept | ✓ | 100.0% | 0.002 |
| `MUL_TEXT_SOCIAL_SPOTIFY_GENRES` | accept | ✓ | 100.0% | 0.005 |
| `MUL_TEXT_TRANSPORTATION_US_ACCIDENTS_MARCH23` | accept | ✓ | 100.0% | 0.01 |
| `REG_TEXT_CONSUMER_BABIES_R_US_PRICES` | accept | ✓ | 100.0% | 0.02 |
| `REG_TEXT_CONSUMER_MERCARI_ONLINE_MARKETPLACE` | accept | ✓ | 100.0% | 0.02 |
| `REG_TEXT_FOOD_ZOMATO_RESTAURANTS` | accept | ✓ | 100.0% | 0.02 |
| `REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER` | accept | ✓ | 100.0% | 0.02 |
| `REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT` | accept | ✓ | 100.0% | 0.01 |
| `REG_TEXT_SOCIAL_BOOK_READABILITY_CLEAR` | accept | ✓ | 100.0% | 0.01 |
| `REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES` | accept | ✓ | 100.0% | 0.02 |
| `REG_TEXT_SOCIAL_VIDEO_GAMES_SALES` | accept | ✓ | 100.0% | 0.05 |
| `MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT` | accept |  | 91.7% | 0.005 |
| `REG_TEXT_CONSUMER_BOOK_PRICE_PREDICTION` | accept | ✓ | 91.7% | 0.01 |
| `MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY` | accept |  | 73.8% | 0.005 |
| `REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY` | accept | ✓ | 73.8% | 0.005 |
| `BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT` | accept |  | 50.0% | 0.002 |
| `BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING` | accept | ✓ | 50.0% | 0.002 |
| `REG_TEXT_FOOD_WINE_VIVINO_SPAIN` | reject |  | 50.0% |  |
| `REG_TEXT_SPORTS_FIFA22_WAGES` | reject |  | 50.0% |  |
| `BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA` | reject |  | 26.2% |  |
| `REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES` | reject |  | 26.2% |  |
| `REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES` | reject |  | 26.2% |  |
| `REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS` | reject |  | 26.2% |  |
| `REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA` | reject |  | 26.2% |  |
| `REG_TEXT_CONSUMER_BIKE_PRICE_BIKEWALE` | reject |  | 8.3% |  |
| `REG_TEXT_CONSUMER_JC_PENNEY_PRODUCT_PRICE` | reject |  | 8.3% |  |
| `REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES` | reject |  | 8.3% |  |
| `REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020` | reject |  | 8.3% |  |
| `REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS` | reject |  | 8.3% |  |
| `REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES` | reject |  | 8.3% |  |
| `REG_TEXT_SOCIAL_KOREAN_DRAMA` | reject |  | 8.3% |  |
| `REG_TEXT_SOCIAL_MUSEUMS_US_REVENUES` | reject |  | 8.3% |  |
| `REG_TEXT_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY` | reject |  | 8.3% |  |
| `REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN` | reject |  | 8.3% |  |
| `BIN_TEXT_SOCIAL_IMDB_GENRE_PREDICTION` | reject |  | 0.0% |  |
| `MUL_TEXT_FOOD_YELP_REVIEWS` | reject |  | 0.0% |  |
| `MUL_TEXT_HOUSES_MELBOURNE_AIRBNB` | reject |  | 0.0% |  |
| `MUL_TEXT_SOCIAL_GOOGLE_QA_TYPE_REASON` | reject |  | 0.0% |  |
| `REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO` | reject |  | 0.0% |  |
| `REG_TEXT_CONSUMER_LAPTOP_INDIAN_PRICES` | reject |  | 0.0% |  |
| `REG_TEXT_FOOD_BEER_RATINGS` | reject |  | 0.0% |  |
| `REG_TEXT_FOOD_COFFEE_REVIEW` | reject |  | 0.0% |  |
| `REG_TEXT_FOOD_RAMEN_RATINGS_2022` | reject |  | 0.0% |  |
| `REG_TEXT_HOUSES_AIRBNB_SEATTLE` | reject |  | 0.0% |  |
| `REG_TEXT_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE` | reject |  | 0.0% |  |
| `REG_TEXT_SOCIAL_ANIME_PLANET_RATING` | reject |  | 0.0% |  |
| `REG_TEXT_SOCIAL_BOOKS_GOODREADS` | reject |  | 0.0% |  |
| `REG_TEXT_SOCIAL_FILMTV_MOVIE_RATING_ITALY` | reject |  | 0.0% |  |
| `REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE` | reject |  | 0.0% |  |
| `REG_TEXT_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT` | reject |  | 0.0% |  |
