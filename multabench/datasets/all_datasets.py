from enum import Enum, unique


@unique
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

    # MulTaBench-Full text datasets: the 20 text extras beyond Core.
    MUL_TEXT_CONSUMER_COMPLAINT = "multabench-full-consumer-complaint"
    MUL_TEXT_BOX_OFFICE = "multabench-full-box-office"
    BIN_TEXT_OSHA_INJURY = "multabench-full-osha-injury"
    MUL_TEXT_NEWS_CHANNEL = "multabench-full-news-channel"
    BIN_TEXT_IMDB_GENRE = "multabench-full-imdb-genre"
    MUL_TEXT_MELBOURNE_AIRBNB = "multabench-full-melbourne-airbnb"
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

    # MulTaBench-Full image extras
    BIN_IMAGE_OASIS_ALZHEIMERS = "multabench-full-oasis-alzheimers"
    MUL_IMAGE_MINECRAFT = "multabench-full-minecraft"
    REG_IMAGE_DVM_CAR = "multabench-full-dvm-car"
    REG_IMAGE_FLIPKART_RATIO = "multabench-full-flipkart-ratio"
    REG_IMAGE_LAHAINA_AUCTION = "multabench-full-lahaina-auction"
    REG_IMAGE_SOCAL_HOUSES = "multabench-full-socal-houses"
    MUL_IMAGE_REDDIT_MEMES = "multabench-full-reddit-memes"
    MUL_IMAGE_POKEMON_HEIGHT = "multabench-full-pokemon-height"
    MUL_IMAGE_PAD_UFES_LESION = "multabench-full-pad-ufes-lesion"
    MUL_IMAGE_HEARTHSTONE_CLASS = "multabench-full-hearthstone-class"
    REG_IMAGE_WATCH_TIER = "multabench-full-watch-tier"
    MUL_IMAGE_HAM10000_LESION = "multabench-full-ham10000-lesion"
    REG_IMAGE_GOIAS_HOUSES = "multabench-full-goias-houses"
    REG_IMAGE_TOKOPEDIA_WEIGHT = "multabench-full-tokopedia-weight"
    REG_IMAGE_KAMERNET_SIZE = "multabench-full-kamernet-size"
    REG_IMAGE_SAO_PAULO_HOUSES = "multabench-full-sao-paulo-houses"
    REG_IMAGE_ROMANIA_PRICE = "multabench-full-romania-price"
    BIN_IMAGE_PINTEREST_POPULAR = "multabench-full-pinterest-popular"
    REG_IMAGE_ZEPTO_PRICE = "multabench-full-zepto-price"
    REG_IMAGE_AIRBNB_NYC = "multabench-full-airbnb-nyc"


_IMAGE_PREFIXES = ("BIN_IMAGE_", "MUL_IMAGE_", "REG_IMAGE_")
_TEXT_PREFIXES = ("BIN_TEXT_", "MUL_TEXT_", "REG_TEXT_")


def is_image_dataset(dataset_id: MulTaBenchDatasetID) -> bool:
    return dataset_id.name.startswith(_IMAGE_PREFIXES)


def is_text_dataset(dataset_id: MulTaBenchDatasetID) -> bool:
    return dataset_id.name.startswith(_TEXT_PREFIXES)


for _d in MulTaBenchDatasetID:
    assert is_image_dataset(_d) != is_text_dataset(_d), f"Dataset {_d.name} must be exactly one of image or text"
