"""The 80 MulTaBench datasets: their IDs, which of them form the benchmark, and where each comes from.

Each dataset is a curated copy of the source dataset in MULTABENCH_SOURCES, hosted on Kaggle under its
MulTaBenchDatasetID slug and, as typed Parquet, on Hugging Face (multabench/datasets/hub.py). Credit
and licensing belong to the source. The curation recipe for each dataset is in
multabench/benchmark/datasets/ at the paper_version tag.
"""
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


# The 40 benchmark datasets are slugged multabench-<name>; the 40 released alongside them,
# admitted on a weaker criterion, are slugged multabench-full-<name>. Their scores must not be
# pooled. Their admission checks are in leaderboard/analysis/ at the paper_version tag.
_RELEASED_ALONGSIDE_PREFIX = "multabench-full-"


def is_benchmark_dataset(dataset_id: MulTaBenchDatasetID) -> bool:
    return not dataset_id.value.startswith(_RELEASED_ALONGSIDE_PREFIX)


MULTABENCH_SOURCES: dict[MulTaBenchDatasetID, str] = {
    MulTaBenchDatasetID.BIN_IMAGE_CELEB_ATTRACTIVENESS: "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset",
    MulTaBenchDatasetID.BIN_IMAGE_HATEFUL_MEME: "https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset",
    MulTaBenchDatasetID.BIN_IMAGE_MAMMOGRAPHY_CMMD: "https://www.kaggle.com/datasets/nguynththanhho/cmmd-mammography",
    MulTaBenchDatasetID.MUL_IMAGE_CBIS_DDSM: "https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset",
    MulTaBenchDatasetID.MUL_IMAGE_CHEXPERT: "https://www.kaggle.com/datasets/ashery/chexpert",
    MulTaBenchDatasetID.MUL_IMAGE_CSGO_SKIN_PRICE: "https://figshare.com/articles/dataset/21454413",
    MulTaBenchDatasetID.MUL_IMAGE_FLOWER_BOUQUETS: "https://www.kaggle.com/datasets/samoilovmikhail/floral-bouquets-images-and-girlfriend-scores",
    MulTaBenchDatasetID.MUL_IMAGE_GLAUCOMA_SMDG: "https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset",
    MulTaBenchDatasetID.MUL_IMAGE_HUBMAP_HPA: "https://www.kaggle.com/datasets/miquel0/hubmaphba-tiled-dataset-512x512",
    MulTaBenchDatasetID.MUL_IMAGE_JUSTIN_INSTAGRAM: "https://www.kaggle.com/datasets/aldiandyainf/which-justin-posted-that",
    MulTaBenchDatasetID.MUL_IMAGE_PETFINDER: "https://www.kaggle.com/c/petfinder-adoption-prediction",
    MulTaBenchDatasetID.MUL_IMAGE_ZOOSCAN_ZOOPLANKTON: "https://www.kaggle.com/datasets/raghavdharwal/pelgass-bay-of-biscay-zooscan-zooplankton-dataset",
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_BEST_SELLER: "https://www.kaggle.com/datasets/amankumar20d/amazon-best-seller-all-departments-us",
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_PACKAGES: "https://www.kaggle.com/datasets/dhruvildave/amazon-bin-image-dataset",
    MulTaBenchDatasetID.REG_IMAGE_HNM_FASHION: "https://www.kaggle.com/datasets/odins0n/handm-dataset-128x128",
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES: "https://www.kaggle.com/datasets/usman8/khaadis-clothes-data-with-images",
    MulTaBenchDatasetID.REG_IMAGE_LETTERBOXD_MOVIES: "https://www.kaggle.com/datasets/gsimonx37/letterboxd",
    MulTaBenchDatasetID.REG_IMAGE_MANGO_MASS: "https://www.kaggle.com/datasets/mypapit/mangomassnet552-dataset",
    MulTaBenchDatasetID.REG_IMAGE_MKPHOTO_BOTS: "https://www.kaggle.com/datasets/guardeec/mkphoto2023",
    MulTaBenchDatasetID.REG_IMAGE_PAINTING_PRICE: "https://www.kaggle.com/datasets/denozavrus/paintings-price-prediction",
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING: "https://www.openml.org/search?type=data&id=46655",
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: "https://www.openml.org/search?type=data&id=46654",
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: "https://www.openml.org/search?type=data&id=46668",
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY: "https://www.openml.org/search?type=data&id=46664",
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS: "https://www.kaggle.com/datasets/ngshiheng/michelin-guide-restaurants-2021",
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT: "https://www.openml.org/search?type=data&id=46651",
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES: "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset",
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: "https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents",
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW: "https://www.openml.org/search?type=data&id=46653",
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW: "https://www.openml.org/search?type=data&id=46659",
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES: "http://pages.cs.wisc.edu/~anhai/data/784_data/baby_products/csv_files/babies_r_us.csv",
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE: "https://www.openml.org/search?type=data&id=46663",
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY: "https://www.kaggle.com/datasets/verracodeguacas/clear-corpus",
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE: "https://www.openml.org/search?type=data&id=46660",
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: "https://www.openml.org/search?type=data&id=42125",
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT: "https://www.scimagojr.com/journalrank.php?out=xls",
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES: "http://pages.cs.wisc.edu/~anhai/data/784_data/movies1/csv_files/rotten_tomatoes.csv",
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES: "https://opendata.vancouver.ca/api/records/1.0/download/?dataset=employee-remuneration-and-expenses-earning-over-75000&format=csv",
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES: "https://www.kaggle.com/datasets/gregorut/videogamesales",
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS: "https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants",
    MulTaBenchDatasetID.MUL_TEXT_CONSUMER_COMPLAINT: "https://www.kaggle.com/selener/consumer-complaint-database",
    MulTaBenchDatasetID.MUL_TEXT_BOX_OFFICE: "https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset",
    MulTaBenchDatasetID.BIN_TEXT_OSHA_INJURY: "https://www.kaggle.com/ruqaiyaship/osha-accident-and-injury-data-1517",
    MulTaBenchDatasetID.MUL_TEXT_NEWS_CHANNEL: "https://www.openml.org/search?type=data&id=46652",
    MulTaBenchDatasetID.BIN_TEXT_IMDB_GENRE: "https://www.openml.org/search?type=data&id=46667",
    MulTaBenchDatasetID.MUL_TEXT_MELBOURNE_AIRBNB: "https://www.openml.org/search?type=data&id=46665",
    MulTaBenchDatasetID.BIN_TEXT_CALIFORNIA_PRICES: "https://www.openml.org/search?type=data&id=46669",
    MulTaBenchDatasetID.MUL_TEXT_BOOKS_GOODREADS: "http://pages.cs.wisc.edu/~anhai/data/784_data/books2/csv_files/goodreads.csv",
    MulTaBenchDatasetID.MUL_TEXT_AMERICAN_EAGLE_PRICES: "https://www.openml.org/search?type=data&id=46656",
    MulTaBenchDatasetID.MUL_TEXT_KOREAN_DRAMA: "https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500",
    MulTaBenchDatasetID.REG_TEXT_WIKILIQ_PRICES: "https://www.kaggle.com/datasets/limtis/wikiliq-dataset",
    MulTaBenchDatasetID.REG_TEXT_CHOCOLATE_BAR_RATINGS: "https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings",
    MulTaBenchDatasetID.REG_TEXT_RAMEN_RATINGS: "https://www.kaggle.com/datasets/ankanhore545/top-ramen-ratings-2022",
    MulTaBenchDatasetID.REG_TEXT_WINE_POLISH_MARKET: "https://www.kaggle.com/datasets/skamlo/wine-price-on-polish-market",
    MulTaBenchDatasetID.REG_TEXT_WINE_VIVINO_SPAIN: "https://www.kaggle.com/datasets/joshuakalobbowles/vivino-wine-data",
    MulTaBenchDatasetID.REG_TEXT_AIRBNB_SEATTLE: "https://www.kaggle.com/datasets/airbnb/seattle",
    MulTaBenchDatasetID.REG_TEXT_ANIME_PLANET: "https://www.kaggle.com/datasets/hernan4444/animeplanet-recommendation-database-2020",
    MulTaBenchDatasetID.REG_TEXT_USED_CAR_PAKISTAN: "https://www.kaggle.com/datasets/mustafaimam/used-car-prices-in-pakistan-2021",
    MulTaBenchDatasetID.REG_TEXT_USED_CAR_SAUDI: "https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset",
    MulTaBenchDatasetID.REG_TEXT_FIFA22_WAGES: "https://www.openml.org/search?type=data&id=45012",
    MulTaBenchDatasetID.BIN_IMAGE_OASIS_ALZHEIMERS: "https://www.kaggle.com/datasets/shreyanmohanty/oasis-alzheimers-detection-multi-class-dataset",
    MulTaBenchDatasetID.MUL_IMAGE_MINECRAFT: "https://www.kaggle.com/datasets/sqdartemy/minecraft-screenshots-dataset-with-features",
    MulTaBenchDatasetID.REG_IMAGE_DVM_CAR: "https://www.kaggle.com/datasets/osamasaifs/dvm-car-with-images",
    MulTaBenchDatasetID.REG_IMAGE_FLIPKART_RATIO: "https://www.kaggle.com/datasets/kuchhbhi/stylish-product-image-dataset",
    MulTaBenchDatasetID.REG_IMAGE_LAHAINA_AUCTION: "https://www.kaggle.com/datasets/quillen/artists-for-lahaina-2023",
    MulTaBenchDatasetID.REG_IMAGE_SOCAL_HOUSES: "https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal",
    MulTaBenchDatasetID.MUL_IMAGE_REDDIT_MEMES: "https://www.kaggle.com/datasets/musadiqpashak/reddit-memes-dataset",
    MulTaBenchDatasetID.MUL_IMAGE_POKEMON_HEIGHT: "https://www.kaggle.com/datasets/divyanshusingh369/complete-pokemon-library-32k-images-and-csv",
    MulTaBenchDatasetID.MUL_IMAGE_PAD_UFES_LESION: "https://www.kaggle.com/datasets/mahdavi1202/skin-cancer",
    MulTaBenchDatasetID.MUL_IMAGE_HEARTHSTONE_CLASS: "https://figshare.com/articles/dataset/21454413",
    MulTaBenchDatasetID.REG_IMAGE_WATCH_TIER: "https://www.kaggle.com/datasets/mathewkouch/a-dataset-of-watches",
    MulTaBenchDatasetID.MUL_IMAGE_HAM10000_LESION: "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000",
    MulTaBenchDatasetID.REG_IMAGE_GOIAS_HOUSES: "https://www.kaggle.com/datasets/carloseduardogo/apartment-prices-in-goinia-gois-brazil",
    MulTaBenchDatasetID.REG_IMAGE_TOKOPEDIA_WEIGHT: "https://www.kaggle.com/datasets/nsmlehq/tokopedia-products-2025",
    MulTaBenchDatasetID.REG_IMAGE_KAMERNET_SIZE: "https://www.kaggle.com/datasets/amancapy/all-kamernet-rent-listings-w-photos-netherlands",
    MulTaBenchDatasetID.REG_IMAGE_SAO_PAULO_HOUSES: "https://www.kaggle.com/datasets/rogerio/image-based-property-price-prediction-in-so-paulo",
    MulTaBenchDatasetID.REG_IMAGE_ROMANIA_PRICE: "https://www.kaggle.com/datasets/furduisorinoctavian/romanian-products-from-emag-and-flanco",
    MulTaBenchDatasetID.BIN_IMAGE_PINTEREST_POPULAR: "https://www.kaggle.com/datasets/andreacombette/pinterest-analysis-using-nlp-and-image-analysis",
    MulTaBenchDatasetID.REG_IMAGE_ZEPTO_PRICE: "https://www.kaggle.com/datasets/simpleaditya/zepto-products-dataset",
    MulTaBenchDatasetID.REG_IMAGE_AIRBNB_NYC: "https://www.kaggle.com/datasets/dominoweir/inside-airbnb-nyc",
}


for _d in MulTaBenchDatasetID:
    assert is_image_dataset(_d) != is_text_dataset(_d), f"Dataset {_d.name} must be exactly one of image or text"
    assert _d in MULTABENCH_SOURCES, f"Dataset {_d.name} has no source"
