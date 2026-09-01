"""
Per-dataset multimodal feature inventory for MulTaBench.

Populated from do_multabench_audit.py (run 2026-04-15).
Keys are MulTaBenchDatasetID members; values are DatasetMultimodalFeatures instances
listing the detected image and free-text columns after curation.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from multabench.datasets.all_datasets import MulTaBenchDatasetID
from multabench.datasets.tiers import MULTABENCH_CORE

NO_PCA_THRESHOLD = 5


@dataclass(frozen=True)
class DatasetMultimodalFeatures:
    image_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.image_cols) + len(self.text_cols)

    @property
    def exceeds_no_pca_threshold(self) -> bool:
        return self.n_total > NO_PCA_THRESHOLD


MULTIMODAL_FEATURES: Dict[MulTaBenchDatasetID, DatasetMultimodalFeatures] = {
    # ── Image datasets ──────────────────────────────────────────────────────
    MulTaBenchDatasetID.BIN_IMAGE_CELEB_ATTRACTIVENESS: DatasetMultimodalFeatures(
        image_cols=["image_id"],
    ),
    MulTaBenchDatasetID.BIN_IMAGE_HATEFUL_MEME: DatasetMultimodalFeatures(
        image_cols=["img"],
    ),
    MulTaBenchDatasetID.BIN_IMAGE_MAMMOGRAPHY_CMMD: DatasetMultimodalFeatures(
        image_cols=["Path"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_CBIS_DDSM: DatasetMultimodalFeatures(
        image_cols=["image file path"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_CHEXPERT: DatasetMultimodalFeatures(
        image_cols=["X-Ray Image"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_CSGO_SKIN_PRICE: DatasetMultimodalFeatures(
        image_cols=["Image Path"],
        text_cols=["Skin Name"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_FLOWER_BOUQUETS: DatasetMultimodalFeatures(
        image_cols=["image_name"],
        text_cols=["description"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_GLAUCOMA_SMDG: DatasetMultimodalFeatures(
        image_cols=["fundus"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_HUBMAP_HPA: DatasetMultimodalFeatures(
        image_cols=["biopsy"],
        text_cols=["rle"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_JUSTIN_INSTAGRAM: DatasetMultimodalFeatures(
        image_cols=["display_picture_url"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_PETFINDER: DatasetMultimodalFeatures(
        image_cols=["Pet Image"],
        text_cols=["Name", "Description", "Breed1", "Breed2"],
    ),
    MulTaBenchDatasetID.MUL_IMAGE_ZOOSCAN_ZOOPLANKTON: DatasetMultimodalFeatures(
        image_cols=["object_image"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_BEST_SELLER: DatasetMultimodalFeatures(
        image_cols=["photo_url"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_PACKAGES: DatasetMultimodalFeatures(
        image_cols=["Amazon bin"],
        text_cols=["product descriptions"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_HNM_FASHION: DatasetMultimodalFeatures(
        image_cols=["ProductPic"],
        text_cols=["detail_desc", "product_type_name", "department_name", "prod_name"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES: DatasetMultimodalFeatures(
        image_cols=["img"],
        text_cols=["Product Description"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_LETTERBOXD_MOVIES: DatasetMultimodalFeatures(
        image_cols=["poster"],
        text_cols=["themes", "name", "tagline"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_MANGO_MASS: DatasetMultimodalFeatures(
        image_cols=["Fruit No"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_MKPHOTO_BOTS: DatasetMultimodalFeatures(
        image_cols=["image"],
    ),
    MulTaBenchDatasetID.REG_IMAGE_PAINTING_PRICE: DatasetMultimodalFeatures(
        image_cols=["image_url"],
        text_cols=["styles", "material"],
    ),
    # ── Text datasets ────────────────────────────────────────────────────────
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING: DatasetMultimodalFeatures(
        text_cols=["description", "salary_range", "title"],
    ),
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY: DatasetMultimodalFeatures(
        text_cols=["created_date", "comment_text"],
    ),
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING: DatasetMultimodalFeatures(
        text_cols=["keywords", "name", "desc", "deadline", "created_at"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY: DatasetMultimodalFeatures(
        text_cols=["job_description", "location", "job_desig", "experience", "key_skills"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS: DatasetMultimodalFeatures(
        text_cols=["Name", "FacilitiesAndServices", "Cuisine", "Location", "Description", "Address"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT: DatasetMultimodalFeatures(
        text_cols=["Product_Description"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES: DatasetMultimodalFeatures(
        text_cols=["artists", "track_name", "album_name"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS: DatasetMultimodalFeatures(
        text_cols=["Weather_Timestamp", "End_Time", "City", "Description", "Start_Time", "County", "Street", "Airport_Code", "Zipcode"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW: DatasetMultimodalFeatures(
        text_cols=["province", "description"],
    ),
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW: DatasetMultimodalFeatures(
        text_cols=["Review Text", "Title"],
    ),
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES: DatasetMultimodalFeatures(
        text_cols=["company_struct", "company_free", "colors", "title"],
    ),
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE: DatasetMultimodalFeatures(
        text_cols=["Title", "Edition", "Author", "Genre", "Synopsis"],
    ),
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY: DatasetMultimodalFeatures(
        text_cols=["URL", "Anthology", "Title", "Author", "British Words", "Excerpt"],
    ),
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE: DatasetMultimodalFeatures(
        text_cols=["cat3", "cat2", "name", "brand_name", "item_description", "category_name"],
    ),
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES: DatasetMultimodalFeatures(
        text_cols=["employee_position_title", "date_first_hired", "full_name", "division"],
    ),
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT: DatasetMultimodalFeatures(
        text_cols=["Publisher", "Ref  / Doc ", "Title", "SJR", "Citations / Doc  (2years)", "Coverage", "Issn", "Areas", "Country", "Categories"],
    ),
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES: DatasetMultimodalFeatures(
        text_cols=["Name", "Language", "Cast", "Filming Locations", "Director", "Description", "Creator", "Actors", "RatingCount", "Genre", "Country", "Release Date", "ReviewCount"],
    ),
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES: DatasetMultimodalFeatures(
        text_cols=["name", "title"],
    ),
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES: DatasetMultimodalFeatures(
        text_cols=["Name", "Publisher"],
    ),
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS: DatasetMultimodalFeatures(
        text_cols=["dish_liked", "name", "cuisines", "address", "reviews_list", "phone", "menu_item"],
    ),
}

_missing_core = [d.name for d in MULTABENCH_CORE if d not in MULTIMODAL_FEATURES]
assert not _missing_core, f"MULTIMODAL_FEATURES is missing MulTaBench-Core datasets: {_missing_core}"

# Uploaded datasets not yet in the inventory -- reported, not asserted, so that uploading a
# Full-tier dataset does not break every import until its features are annotated.
UNAUDITED = [d for d in MulTaBenchDatasetID if d not in MULTIMODAL_FEATURES]
