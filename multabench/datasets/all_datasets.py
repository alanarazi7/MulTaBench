from enum import Enum


class OpenMLDatasetID(Enum):
    BIN_TEXT_SOCIAL_JIGSAW_TOXICITY = 46654
    BIN_TEXT_PROFESSIONAL_KICKSTARTER_FUNDING = 46668
    BIN_TEXT_PROFESSIONAL_FAKE_JOB_POSTING = 46655
    MUL_TEXT_CONSUMER_WOMEN_ECOMMERCE_CLOTHING_REVIEW = 46659
    MUL_TEXT_PROFESSIONAL_DATA_SCIENTIST_SALARY = 46664
    MUL_TEXT_CONSUMER_PRODUCT_SENTIMENT = 46651
    BIN_TEXT_SOCIAL_IMDB_GENRE_PREDICTION = 46667
    MUL_TEXT_FOOD_WINE_REVIEW = 46653
    MUL_TEXT_HOUSES_MELBOURNE_AIRBNB = 46665
    MUL_TEXT_SOCIAL_GOOGLE_QA_TYPE_REASON = 46658
    MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY = 46652
    REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES = 46656
    REG_TEXT_CONSUMER_JC_PENNEY_PRODUCT_PRICE = 46661
    REG_TEXT_PROFESSIONAL_EMPLOYEE_SALARY_MONTGOMERY = 42125
    REG_TEXT_SPORTS_FIFA22_WAGES = 45012
    REG_TEXT_CONSUMER_BOOK_PRICE_PREDICTION = 46663
    REG_TEXT_CONSUMER_MERCARI_ONLINE_MARKETPLACE = 46660
    REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020 = 46669

class KaggleDatasetID(Enum):
    MUL_IMAGE_MNIST_HAM1000_CANCER_SKIN_LESIONS = "kmader/skin-cancer-mnist-ham10000"
    BIN_IMAGE_SOCIAL_TRENDING_BOOKS = "kuchhbhi/treding-book-dataset"
    MUL_IMAGE_STARTUP_2023_COMPANY_EMPLOYEE_SIZES = "chickooo/top-tech-startups-hiring-2023"
    MUL_IMAGE_CONSUMER_WALMART_PRODUCT_PRICING = "hikageshinomori/walmart-e-commerce-product-data"
    MUL_IMAGE_TOKOPEDIA_PRODUCTS_WEIGHT_ESTIMATION = "nsmlehq/tokopedia-products-2025"
    REG_IMAGE_STYLISTIC_PRODUCT_PRICE_RATIO_FLIPKART = "kuchhbhi/stylish-product-image-dataset"
    # BagOfTricks image datasets
    MUL_IMAGE_ARTM_ART_PRICE_DATASET_MOVEMENT = "flkuhm/art-price-dataset"
    REG_IMAGE_DVM_DEEP_VISUAL_MARKETING_CAR_PRICES = "osamasaifs/dvm-car-with-images"
    REG_IMAGE_GOODREADS_BOOKS_RATING = "mdhamani/goodreads-books-100k"
    MUL_IMAGE_PETFINDER_ADOPTION_SPEED = "c/petfinder-adoption-prediction"
    MUL_IMAGE_HEALTHCARE_COVID_CHESTXRAY_PNEUMONIA = "bachrr/covid-chest-xray"
    REG_IMAGE_HOUSES_AIRBNB_SEATTLE = "airbnb/seattle/"
    # Text data
    BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT = "selener/consumer-complaint-database"
    BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA = "ruqaiyaship/osha-accident-and-injury-data-1517"
    MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT = "jeradrose/hearthstone-cards"
    REG_TEXT_CONSUMER_LAPTOP_INDIAN_PRICES = "dhanushbommavaram/laptop-dataset"
    REG_TEXT_HOUSES_SAN_FRANCISCO_PERMITS_APPLICATIONS = "aparnashastry/building-permit-applications-data"
    MUL_TEXT_FOOD_MICHELIN_GUIDE_RESTAURANTS = "ngshiheng/michelin-guide-restaurants-2021"
    MUL_TEXT_FOOD_YELP_REVIEWS = "omkarsabnis/yelp-reviews-dataset"
    REG_TEXT_FOOD_ZOMATO_RESTAURANTS = "himanshupoddar/zomato-bangalore-restaurants"
    MUL_TEXT_SOCIAL_SPOTIFY_GENRES = "maharshipandya/-spotify-tracks-dataset"
    MUL_TEXT_TRANSPORTATION_US_ACCIDENTS_MARCH23 = "sobhanmoosavi/us-accidents"
    REG_TEXT_CONSUMER_CAR_PRICE_CARDEKHO = "sukritchatterjee/used-cars-dataset-cardekho"
    REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES = "limtis/wikiliq-dataset"
    REG_TEXT_FOOD_BEER_RATINGS = "ruthgn/beer-profile-and-ratings-data-set"
    REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS = "rtatman/chocolate-bar-ratings"
    REG_TEXT_FOOD_COFFEE_REVIEW = "hanifalirsyad/coffee-scrap-coffeereview"
    REG_TEXT_FOOD_RAMEN_RATINGS_2022 = "ankanhore545/top-ramen-ratings-2022"
    REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES = "skamlo/wine-price-on-polish-market"
    REG_TEXT_FOOD_WINE_VIVINO_SPAIN = "joshuakalobbowles/vivino-wine-data"
    REG_TEXT_HOUSES_AIRBNB_SEATTLE = "airbnb/seattle"
    REG_TEXT_PROFESSIONAL_COMPANY_EMPLOYEES_SIZE = "peopledatalabssf/free-7-million-company-dataset"
    REG_TEXT_SOCIAL_ANIME_PLANET_RATING = "hernan4444/animeplanet-recommendation-database-2020"
    REG_TEXT_SOCIAL_BOOK_READABILITY_CLEAR = "verracodeguacas/clear-corpus"
    REG_TEXT_SOCIAL_FILMTV_MOVIE_RATING_ITALY = "stefanoleone992/filmtv-movies-dataset"
    REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE = "rounakbanik/the-movies-dataset"
    REG_TEXT_SOCIAL_MUSEUMS_US_REVENUES = "markusschmitz/museums"
    REG_TEXT_SOCIAL_VIDEO_GAMES_SALES = "gregorut/videogamesales"
    REG_TEXT_SPORTS_NBA_DRAFT_VALUE_OVER_REPLACEMENT = "mattop/nba-draft-basketball-player-data-19892021"
    REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN = "mustafaimam/used-car-prices-in-pakistan-2021"
    REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA = "turkibintalib/saudi-arabia-used-cars-dataset"
    REG_TEXT_TRANSPORTATION_USED_CAR_MERCEDES_BENZ_ITALY = "bogdansorin/second-hand-mercedes-benz-registered-2000-2023-ita"
    REG_TEXT_SOCIAL_KOREAN_DRAMA = "noorrizki/top-korean-drama-list-1500"

class MulTaBenchDatasetID(Enum):
    BIN_IMAGE_CELEB_ATTRACTIVENESS = "multabench-celeb-attractiveness"
    BIN_IMAGE_HATEFUL_MEME = "multabench-hateful-meme"
    BIN_IMAGE_MAMMOGRAPHY_CMMD = "multabench-mammography-cmmd"
    MUL_IMAGE_CBIS_DDSM = "multabench-cbis-ddsm"
    MUL_IMAGE_CHEXPERT = "multabench-chexpert"
    MUL_IMAGE_CSGO_SKIN_PRICE = "multabench-csgo-skin"
    MUL_IMAGE_FLOWER_BOUQUETS = "multabench-flower-bouquets"
    MUL_IMAGE_GLAUCOMA_SMDG = "multabench-glaucoma-smdg"
    MUL_IMAGE_HUBMAP_HPA = "multabench-hubmap-hpa"
    MUL_IMAGE_JUSTIN_INSTAGRAM = "multabench-justin-instagram"
    MUL_IMAGE_PETFINDER = "multabench-petfinder"
    MUL_IMAGE_ZOOSCAN_ZOOPLANKTON = "multabench-zooscan-zooplankton"
    REG_IMAGE_AMAZON_BEST_SELLER = "multabench-amazon-bestseller"
    REG_IMAGE_AMAZON_PACKAGES = "multabench-amazon-packages"
    REG_IMAGE_HNM_FASHION = "multabench-hnm-fashion"
    REG_IMAGE_KHAADI_CLOTHES = "multabench-khaadi-clothes"
    REG_IMAGE_LETTERBOXD_MOVIES = "multabench-letterboxd-movies"
    REG_IMAGE_MANGO_MASS = "multabench-mango-mass"
    REG_IMAGE_MKPHOTO_BOTS = "multabench-mkphoto-bots"
    REG_IMAGE_PAINTING_PRICE = "multabench-painting-price"
    # Text datasets
    BIN_TEXT_FAKE_JOB_POSTING = "multabench-fake-job-posting"
    BIN_TEXT_JIGSAW_TOXICITY = "multabench-jigsaw-toxicity"
    BIN_TEXT_KICKSTARTER_FUNDING = "multabench-kickstarter-funding"
    MUL_TEXT_DATA_SCIENTIST_SALARY = "multabench-data-scientist-salary"
    MUL_TEXT_MICHELIN_RESTAURANTS = "multabench-michelin-restaurants"
    MUL_TEXT_PRODUCT_SENTIMENT = "multabench-product-sentiment"
    MUL_TEXT_SPOTIFY_GENRES = "multabench-spotify-genres"
    MUL_TEXT_US_ACCIDENTS = "multabench-us-accidents"
    MUL_TEXT_WINE_REVIEW = "multabench-wine-review"
    MUL_TEXT_WOMEN_CLOTHING_REVIEW = "multabench-women-clothing-review"
    REG_TEXT_BABIES_PRICES = "multabench-babies-prices"
    REG_TEXT_BOOK_PRICE = "multabench-book-price"
    REG_TEXT_BOOK_READABILITY = "multabench-book-readability"
    REG_TEXT_MERCARI_MARKETPLACE = "multabench-mercari-marketplace"
    REG_TEXT_MONTGOMERY_SALARIES = "multabench-montgomery-salaries"
    REG_TEXT_SCIMAGOJR_IMPACT = "multabench-scimagojr-impact"
    REG_TEXT_ROTTEN_TOMATOES = "multabench-rotten-tomatoes"
    REG_TEXT_VANCOUVER_SALARIES = "multabench-vancouver-salaries"
    REG_TEXT_VIDEO_GAMES_SALES = "multabench-video-games-sales"
    REG_TEXT_ZOMATO_RESTAURANTS = "multabench-zomato-restaurants"

    # MulTaBench-Full additions: classification datasets from the 56-dataset text pool that pass
    # Joint Signal (they were kept out of Core because fine-tuning the encoder adds nothing).
    MUL_TEXT_CONSUMER_COMPLAINT = "multabench-full-consumer-complaint"
    MUL_TEXT_HEARTHSTONE_CARDS = "multabench-full-hearthstone-cards"
    BIN_TEXT_OSHA_INJURY = "multabench-full-osha-injury"
    MUL_TEXT_NEWS_CHANNEL = "multabench-full-news-channel"
    BIN_TEXT_IMDB_GENRE = "multabench-full-imdb-genre"
    MUL_TEXT_MELBOURNE_AIRBNB = "multabench-full-melbourne-airbnb"

    # MulTaBench-Full: the 14 remaining text extras, completing the text half at 40. Four carry a
    # target reformulated from regression into equal-frequency bins at curation time -- which is
    # why their names are BIN_/MUL_ while their sources are REG_; each recipe's TARGET_BINS holds
    # its bin count. Ten stay regression.
    BIN_TEXT_CALIFORNIA_PRICES = "multabench-full-california-prices"
    MUL_TEXT_BOOKS_GOODREADS = "multabench-full-books-goodreads"
    MUL_TEXT_AMERICAN_EAGLE_PRICES = "multabench-full-american-eagle-prices"
    MUL_TEXT_KOREAN_DRAMA = "multabench-full-korean-drama"
    REG_TEXT_WIKILIQ_PRICES = "multabench-full-wikiliq-prices"
    REG_TEXT_CHOCOLATE_BAR_RATINGS = "multabench-full-chocolate-bar-ratings"
    REG_TEXT_RAMEN_RATINGS = "multabench-full-ramen-ratings"
    REG_TEXT_WINE_POLISH_MARKET = "multabench-full-wine-polish-market"
    REG_TEXT_WINE_VIVINO_SPAIN = "multabench-full-wine-vivino-spain"
    REG_TEXT_AIRBNB_SEATTLE = "multabench-full-airbnb-seattle"
    REG_TEXT_ANIME_PLANET = "multabench-full-anime-planet"
    REG_TEXT_USED_CAR_PAKISTAN = "multabench-full-used-car-pakistan"
    REG_TEXT_USED_CAR_SAUDI = "multabench-full-used-car-saudi"
    REG_TEXT_FIFA22_WAGES = "multabench-full-fifa22-wages"


class UrlDatasetID(Enum):
    # Image Benchmark - Candidates
    MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY = "https://figshare.com/ndownloader/files/38077608"
    # Text data
    REG_TEXT_CONSUMER_BABIES_R_US_PRICES = "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv"
    REG_TEXT_CONSUMER_BIKE_PRICE_BIKEWALE = "http://pages.cs.wisc.edu/~anhai/data/784_data/bikes/csv_files/bikewale.csv"
    REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER = "https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv"
    REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES = "https://ai-jobs.net/salaries/download/salaries.csv"
    REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT = "https://www.scimagojr.com/journalrank.php?out=xls"
    REG_TEXT_SOCIAL_BOOKS_GOODREADS = "http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv"
    REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES = "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv"


MultimodalDatasetID = KaggleDatasetID | UrlDatasetID | OpenMLDatasetID | MulTaBenchDatasetID

ALL_DATASETS = list(MulTaBenchDatasetID) + list(KaggleDatasetID) + list(UrlDatasetID) + list(OpenMLDatasetID)

_IMAGE_PREFIXES = ("BIN_IMAGE_", "MUL_IMAGE_", "REG_IMAGE_")
_TEXT_PREFIXES = ("BIN_TEXT_", "MUL_TEXT_", "REG_TEXT_")


def is_image_dataset(dataset_id: MultimodalDatasetID) -> bool:
    return dataset_id.name.startswith(_IMAGE_PREFIXES)


def is_text_dataset(dataset_id: MultimodalDatasetID) -> bool:
    return dataset_id.name.startswith(_TEXT_PREFIXES)

ALL_IMAGE_DATASETS = [d for d in ALL_DATASETS if is_image_dataset(d)]


for _d in ALL_DATASETS:
    assert is_image_dataset(_d) or is_text_dataset(_d), f"Dataset {_d.name} is neither image nor text"
    assert not (is_image_dataset(_d) and is_text_dataset(_d)), f"Dataset {_d.name} is both image and text"

_all_names = [d.name for d in ALL_DATASETS]
_all_values = [d.value for d in ALL_DATASETS]
_dup_names = sorted({n for n in _all_names if _all_names.count(n) > 1})
_dup_values = sorted({str(v) for v in _all_values if _all_values.count(v) > 1})
if _dup_names:
    raise ValueError(f"Duplicate dataset names across enums: {_dup_names}")
if _dup_values:
    raise ValueError(f"Duplicate dataset values across enums: {_dup_values}")