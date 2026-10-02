"""
MulTaBench datasets on the Hugging Face Hub: one dataset repo each, holding a typed data.parquet and metadata.json.

Image datasets also hold their images packed in images-*.zip shards (the Hub limits files per folder and per repo).
Their paths inside the shards are the paths in the image column, so extracting the shards next to data.parquet
restores the images/ folder.
"""
import os
import tempfile
import zipfile
from glob import glob
from os.path import join

from multabench.datasets.all_datasets import MulTaBenchDatasetID

HF_ORG = "multabench"
DATA_PARQUET = "data.parquet"
IMAGES_DIR = "images"
IMAGE_SHARDS = "images-*.zip"


# Repo names are <core|extended>-<img|text>-<reg|cls>-<name>: core for the benchmark datasets, extended for the ones
# released alongside them.
HF_REPOS: dict[MulTaBenchDatasetID, str] = {
    MulTaBenchDatasetID.BIN_IMAGE_CELEB_ATTRACTIVENESS: "core-img-cls-celeb-attractiveness",
    MulTaBenchDatasetID.BIN_IMAGE_HATEFUL_MEME: "core-img-cls-hateful-meme",
    MulTaBenchDatasetID.BIN_IMAGE_MAMMOGRAPHY_CMMD: "core-img-cls-mammography-cmmd",
    MulTaBenchDatasetID.MUL_IMAGE_CBIS_DDSM: "core-img-cls-cbis-ddsm",
    MulTaBenchDatasetID.MUL_IMAGE_CHEXPERT: "core-img-cls-chexpert",
    MulTaBenchDatasetID.MUL_IMAGE_CSGO_SKIN_PRICE: "core-img-cls-csgo-skin",
    MulTaBenchDatasetID.MUL_IMAGE_FLOWER_BOUQUETS: "core-img-cls-flower-bouquets",
    MulTaBenchDatasetID.MUL_IMAGE_GLAUCOMA_SMDG: "core-img-cls-glaucoma-smdg",
    MulTaBenchDatasetID.MUL_IMAGE_HUBMAP_HPA: "core-img-cls-hubmap-hpa",
    MulTaBenchDatasetID.MUL_IMAGE_JUSTIN_INSTAGRAM: "core-img-cls-justin-instagram",
    MulTaBenchDatasetID.MUL_IMAGE_PETFINDER: "core-img-cls-petfinder",
    MulTaBenchDatasetID.MUL_IMAGE_ZOOSCAN_ZOOPLANKTON: "core-img-cls-zooscan-zooplankton",
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_BEST_SELLER: "core-img-reg-amazon-bestseller",
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_PACKAGES: "core-img-reg-amazon-packages",
    MulTaBenchDatasetID.REG_IMAGE_HNM_FASHION: "core-img-reg-hnm-fashion",
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES: "core-img-reg-khaadi-clothes",
    MulTaBenchDatasetID.REG_IMAGE_LETTERBOXD_MOVIES: "core-img-reg-letterboxd-movies",
    MulTaBenchDatasetID.REG_IMAGE_MANGO_MASS: "core-img-reg-mango-mass",
    MulTaBenchDatasetID.REG_IMAGE_MKPHOTO_BOTS: "core-img-reg-mkphoto-bots",
    MulTaBenchDatasetID.REG_IMAGE_PAINTING_PRICE: "core-img-reg-painting-price",
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING: "core-text-cls-fake-job-posting",
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: "core-text-cls-jigsaw-toxicity",
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: "core-text-cls-kickstarter-funding",
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY: "core-text-cls-data-scientist-salary",
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS: "core-text-cls-michelin-restaurants",
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT: "core-text-cls-product-sentiment",
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES: "core-text-cls-spotify-genres",
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: "core-text-cls-us-accidents",
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW: "core-text-cls-wine-review",
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW: "core-text-cls-women-clothing-review",
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES: "core-text-reg-babies-prices",
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE: "core-text-reg-book-price",
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY: "core-text-reg-book-readability",
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE: "core-text-reg-mercari-marketplace",
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: "core-text-reg-montgomery-salaries",
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT: "core-text-reg-scimagojr-impact",
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES: "core-text-reg-rotten-tomatoes",
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES: "core-text-reg-vancouver-salaries",
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES: "core-text-reg-video-games-sales",
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS: "core-text-reg-zomato-restaurants",
    MulTaBenchDatasetID.MUL_TEXT_CONSUMER_COMPLAINT: "extended-text-cls-consumer-complaint",
    MulTaBenchDatasetID.MUL_TEXT_BOX_OFFICE: "extended-text-cls-box-office",
    MulTaBenchDatasetID.BIN_TEXT_OSHA_INJURY: "extended-text-cls-osha-injury",
    MulTaBenchDatasetID.MUL_TEXT_NEWS_CHANNEL: "extended-text-cls-news-channel",
    MulTaBenchDatasetID.BIN_TEXT_IMDB_GENRE: "extended-text-cls-imdb-genre",
    MulTaBenchDatasetID.MUL_TEXT_MELBOURNE_AIRBNB: "extended-text-cls-melbourne-airbnb",
    MulTaBenchDatasetID.BIN_TEXT_CALIFORNIA_PRICES: "extended-text-cls-california-prices",
    MulTaBenchDatasetID.MUL_TEXT_BOOKS_GOODREADS: "extended-text-cls-books-goodreads",
    MulTaBenchDatasetID.MUL_TEXT_AMERICAN_EAGLE_PRICES: "extended-text-cls-american-eagle-prices",
    MulTaBenchDatasetID.MUL_TEXT_KOREAN_DRAMA: "extended-text-cls-korean-drama",
    MulTaBenchDatasetID.REG_TEXT_WIKILIQ_PRICES: "extended-text-reg-wikiliq-prices",
    MulTaBenchDatasetID.REG_TEXT_CHOCOLATE_BAR_RATINGS: "extended-text-reg-chocolate-bar-ratings",
    MulTaBenchDatasetID.REG_TEXT_RAMEN_RATINGS: "extended-text-reg-ramen-ratings",
    MulTaBenchDatasetID.REG_TEXT_WINE_POLISH_MARKET: "extended-text-reg-wine-polish-market",
    MulTaBenchDatasetID.REG_TEXT_WINE_VIVINO_SPAIN: "extended-text-reg-wine-vivino-spain",
    MulTaBenchDatasetID.REG_TEXT_AIRBNB_SEATTLE: "extended-text-reg-airbnb-seattle",
    MulTaBenchDatasetID.REG_TEXT_ANIME_PLANET: "extended-text-reg-anime-planet",
    MulTaBenchDatasetID.REG_TEXT_USED_CAR_PAKISTAN: "extended-text-reg-used-car-pakistan",
    MulTaBenchDatasetID.REG_TEXT_USED_CAR_SAUDI: "extended-text-reg-used-car-saudi",
    MulTaBenchDatasetID.REG_TEXT_FIFA22_WAGES: "extended-text-reg-fifa22-wages",
    MulTaBenchDatasetID.BIN_IMAGE_OASIS_ALZHEIMERS: "extended-img-cls-oasis-alzheimers",
    MulTaBenchDatasetID.MUL_IMAGE_MINECRAFT: "extended-img-cls-minecraft",
    MulTaBenchDatasetID.REG_IMAGE_DVM_CAR: "extended-img-reg-dvm-car",
    MulTaBenchDatasetID.REG_IMAGE_FLIPKART_RATIO: "extended-img-reg-flipkart-ratio",
    MulTaBenchDatasetID.REG_IMAGE_LAHAINA_AUCTION: "extended-img-reg-lahaina-auction",
    MulTaBenchDatasetID.REG_IMAGE_SOCAL_HOUSES: "extended-img-reg-socal-houses",
    MulTaBenchDatasetID.MUL_IMAGE_REDDIT_MEMES: "extended-img-cls-reddit-memes",
    MulTaBenchDatasetID.MUL_IMAGE_POKEMON_HEIGHT: "extended-img-cls-pokemon-height",
    MulTaBenchDatasetID.MUL_IMAGE_PAD_UFES_LESION: "extended-img-cls-pad-ufes-lesion",
    MulTaBenchDatasetID.MUL_IMAGE_HEARTHSTONE_CLASS: "extended-img-cls-hearthstone-class",
    MulTaBenchDatasetID.REG_IMAGE_WATCH_TIER: "extended-img-reg-watch-tier",
    MulTaBenchDatasetID.MUL_IMAGE_HAM10000_LESION: "extended-img-cls-ham10000-lesion",
    MulTaBenchDatasetID.REG_IMAGE_GOIAS_HOUSES: "extended-img-reg-goias-houses",
    MulTaBenchDatasetID.REG_IMAGE_TOKOPEDIA_WEIGHT: "extended-img-reg-tokopedia-weight",
    MulTaBenchDatasetID.REG_IMAGE_KAMERNET_SIZE: "extended-img-reg-kamernet-size",
    MulTaBenchDatasetID.REG_IMAGE_SAO_PAULO_HOUSES: "extended-img-reg-sao-paulo-houses",
    MulTaBenchDatasetID.REG_IMAGE_ROMANIA_PRICE: "extended-img-reg-romania-price",
    MulTaBenchDatasetID.BIN_IMAGE_PINTEREST_POPULAR: "extended-img-cls-pinterest-popular",
    MulTaBenchDatasetID.REG_IMAGE_ZEPTO_PRICE: "extended-img-reg-zepto-price",
    MulTaBenchDatasetID.REG_IMAGE_AIRBNB_NYC: "extended-img-reg-airbnb-nyc",
}


def hf_repo_id(dataset_id: MulTaBenchDatasetID) -> str:
    return f"{HF_ORG}/{HF_REPOS[dataset_id]}"


def extract_images(dir_path: str):
    images_dir = join(dir_path, IMAGES_DIR)
    shards = sorted(glob(join(dir_path, IMAGE_SHARDS)))
    if not shards or os.path.isdir(images_dir):
        return
    print(f"Extracting {len(shards)} image shards...")
    tmp_dir = tempfile.mkdtemp(dir=dir_path)
    for shard in shards:
        with zipfile.ZipFile(shard) as z:
            z.extractall(tmp_dir)
    os.rename(join(tmp_dir, IMAGES_DIR), images_dir)
    os.rmdir(tmp_dir)
