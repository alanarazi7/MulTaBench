"""MulTaBench tiering: Core and Full.

Core is the benchmark of the published paper, 20 image-tabular and 20 text-tabular datasets that
show both Joint Signal and Task-awareness. Full relaxes admission to Joint Signal alone and
expands each half to 40. Both rules are defined over the curation deltas in
`multabench.leaderboard.analysis.pass_matrix`.

Membership is typed over `MultimodalDatasetID`, so a Full member can load from its original source
before it is re-hosted as a curated `multabench-*` dataset; `PROMOTED_FROM` records the promotion.
"""
from enum import Enum
from typing import Dict, List, Literal, Optional

from multabench.datasets.all_datasets import (
    KaggleDatasetID,
    MulTaBenchDatasetID,
    MultimodalDatasetID,
    OpenMLDatasetID,
    UrlDatasetID,
    is_image_dataset,
    is_text_dataset,
)


class Tier(Enum):
    CORE = "core"
    FULL = "full"


Modality = Literal["image", "text"]

CORE_PER_MODALITY = 20
FULL_PER_MODALITY = 40


# Order matters: it fixes the row order of the paper tables and datasets_summary.csv.
MULTABENCH_CORE_IMAGE: List[MultimodalDatasetID] = [
    MulTaBenchDatasetID.BIN_IMAGE_CELEB_ATTRACTIVENESS,
    MulTaBenchDatasetID.BIN_IMAGE_HATEFUL_MEME,
    MulTaBenchDatasetID.BIN_IMAGE_MAMMOGRAPHY_CMMD,
    MulTaBenchDatasetID.MUL_IMAGE_CHEXPERT,
    MulTaBenchDatasetID.MUL_IMAGE_CBIS_DDSM,
    MulTaBenchDatasetID.MUL_IMAGE_GLAUCOMA_SMDG,
    MulTaBenchDatasetID.MUL_IMAGE_CSGO_SKIN_PRICE,
    MulTaBenchDatasetID.MUL_IMAGE_FLOWER_BOUQUETS,
    MulTaBenchDatasetID.MUL_IMAGE_HUBMAP_HPA,
    MulTaBenchDatasetID.MUL_IMAGE_JUSTIN_INSTAGRAM,
    MulTaBenchDatasetID.MUL_IMAGE_PETFINDER,
    MulTaBenchDatasetID.MUL_IMAGE_ZOOSCAN_ZOOPLANKTON,
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_BEST_SELLER,
    MulTaBenchDatasetID.REG_IMAGE_AMAZON_PACKAGES,
    MulTaBenchDatasetID.REG_IMAGE_HNM_FASHION,
    MulTaBenchDatasetID.REG_IMAGE_KHAADI_CLOTHES,
    MulTaBenchDatasetID.REG_IMAGE_LETTERBOXD_MOVIES,
    MulTaBenchDatasetID.REG_IMAGE_MANGO_MASS,
    MulTaBenchDatasetID.REG_IMAGE_MKPHOTO_BOTS,
    MulTaBenchDatasetID.REG_IMAGE_PAINTING_PRICE,
]

MULTABENCH_CORE_TEXT: List[MultimodalDatasetID] = [
    MulTaBenchDatasetID.BIN_TEXT_FAKE_JOB_POSTING,
    MulTaBenchDatasetID.BIN_TEXT_JIGSAW_TOXICITY,
    MulTaBenchDatasetID.BIN_TEXT_KICKSTARTER_FUNDING,
    MulTaBenchDatasetID.MUL_TEXT_DATA_SCIENTIST_SALARY,
    MulTaBenchDatasetID.MUL_TEXT_MICHELIN_RESTAURANTS,
    MulTaBenchDatasetID.MUL_TEXT_PRODUCT_SENTIMENT,
    MulTaBenchDatasetID.MUL_TEXT_SPOTIFY_GENRES,
    MulTaBenchDatasetID.MUL_TEXT_US_ACCIDENTS,
    MulTaBenchDatasetID.MUL_TEXT_WINE_REVIEW,
    MulTaBenchDatasetID.MUL_TEXT_WOMEN_CLOTHING_REVIEW,
    MulTaBenchDatasetID.REG_TEXT_BABIES_PRICES,
    MulTaBenchDatasetID.REG_TEXT_BOOK_PRICE,
    MulTaBenchDatasetID.REG_TEXT_BOOK_READABILITY,
    MulTaBenchDatasetID.REG_TEXT_MERCARI_MARKETPLACE,
    MulTaBenchDatasetID.REG_TEXT_MONTGOMERY_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_ROTTEN_TOMATOES,
    MulTaBenchDatasetID.REG_TEXT_SCIMAGOJR_IMPACT,
    MulTaBenchDatasetID.REG_TEXT_VANCOUVER_SALARIES,
    MulTaBenchDatasetID.REG_TEXT_VIDEO_GAMES_SALES,
    MulTaBenchDatasetID.REG_TEXT_ZOMATO_RESTAURANTS,
]

# Full-only datasets, on top of Core.
MULTABENCH_FULL_IMAGE_EXTRA: List[MultimodalDatasetID] = [
]

MULTABENCH_FULL_TEXT_EXTRA: List[MultimodalDatasetID] = [
    KaggleDatasetID.REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES,
    KaggleDatasetID.REG_TEXT_SOCIAL_KOREAN_DRAMA,
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA,
    KaggleDatasetID.BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT,
    KaggleDatasetID.MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT,
    KaggleDatasetID.REG_TEXT_FOOD_WINE_VIVINO_SPAIN,
    KaggleDatasetID.REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS,
    KaggleDatasetID.REG_TEXT_SOCIAL_ANIME_PLANET_RATING,
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN,
    UrlDatasetID.REG_TEXT_SOCIAL_BOOKS_GOODREADS,
    KaggleDatasetID.REG_TEXT_FOOD_RAMEN_RATINGS_2022,
    KaggleDatasetID.BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA,
    OpenMLDatasetID.REG_TEXT_SPORTS_FIFA22_WAGES,
    OpenMLDatasetID.REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020,
    KaggleDatasetID.REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES,
    OpenMLDatasetID.REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES,
    KaggleDatasetID.REG_TEXT_HOUSES_AIRBNB_SEATTLE,
    OpenMLDatasetID.MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY,
    KaggleDatasetID.REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE,
    UrlDatasetID.REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,
]

MULTABENCH_CORE: List[MultimodalDatasetID] = MULTABENCH_CORE_IMAGE + MULTABENCH_CORE_TEXT
MULTABENCH_FULL_IMAGE: List[MultimodalDatasetID] = MULTABENCH_CORE_IMAGE + MULTABENCH_FULL_IMAGE_EXTRA
MULTABENCH_FULL_TEXT: List[MultimodalDatasetID] = MULTABENCH_CORE_TEXT + MULTABENCH_FULL_TEXT_EXTRA
MULTABENCH_FULL: List[MultimodalDatasetID] = MULTABENCH_FULL_IMAGE + MULTABENCH_FULL_TEXT

# Curated id -> the source id it was promoted from; the names do not reveal it.
PROMOTED_FROM: Dict[MulTaBenchDatasetID, MultimodalDatasetID] = {}

_TIER_LISTS = {Tier.CORE: MULTABENCH_CORE, Tier.FULL: MULTABENCH_FULL}
_TIER_SETS = {tier: frozenset(datasets) for tier, datasets in _TIER_LISTS.items()}
_MODALITY_LISTS = {
    (Tier.CORE, "image"): MULTABENCH_CORE_IMAGE,
    (Tier.CORE, "text"): MULTABENCH_CORE_TEXT,
    (Tier.FULL, "image"): MULTABENCH_FULL_IMAGE,
    (Tier.FULL, "text"): MULTABENCH_FULL_TEXT,
}


def _as_tier(tier: "Tier | str") -> Tier:
    if isinstance(tier, Tier):
        return tier
    return tier_from_name(str(tier))


def tier_from_name(name: str) -> Tier:
    try:
        return Tier(name.strip().lower())
    except ValueError:
        raise ValueError(f"Unknown tier '{name}'. Choose from {[t.value for t in Tier]}")


def datasets_for_tier(tier: "Tier | str", modality: Optional[Modality] = None) -> List[MultimodalDatasetID]:
    """The datasets of a tier, in canonical order (image half first, then text half)."""
    tier = _as_tier(tier)
    if modality is None:
        return list(_TIER_LISTS[tier])
    if modality not in ("image", "text"):
        raise ValueError(f"Unknown modality '{modality}'. Choose from ['image', 'text']")
    return list(_MODALITY_LISTS[(tier, modality)])


def is_in_tier(dataset_id: MultimodalDatasetID, tier: "Tier | str") -> bool:
    """Core is a subset of Full, so a Core dataset is in Full too."""
    return dataset_id in _TIER_SETS[_as_tier(tier)]


def get_tier(dataset_id: MultimodalDatasetID) -> Optional[Tier]:
    """The narrowest tier containing this dataset, or None if it is in no tier."""
    for tier in (Tier.CORE, Tier.FULL):
        if is_in_tier(dataset_id, tier):
            return tier
    return None


def is_curated(dataset_id: MultimodalDatasetID) -> bool:
    return isinstance(dataset_id, MulTaBenchDatasetID)


def pending_upload(tier: "Tier | str" = Tier.FULL) -> List[MultimodalDatasetID]:
    """Tier members that still load through their original source instead of the unified API."""
    return [d for d in datasets_for_tier(tier) if not is_curated(d)]


def untiered_curated() -> List[MulTaBenchDatasetID]:
    """Uploaded datasets in no tier: drift, not an error."""
    return [d for d in MulTaBenchDatasetID if get_tier(d) is None]


def _assert_no_duplicates(datasets: List[MultimodalDatasetID], label: str) -> None:
    seen = set()
    duplicated = [d.name for d in datasets if d in seen or seen.add(d)]
    assert not duplicated, f"{label} contains duplicate datasets: {duplicated}"


_assert_no_duplicates(MULTABENCH_FULL, "MULTABENCH_FULL")

assert len(MULTABENCH_CORE_IMAGE) == CORE_PER_MODALITY, f"Core image half is {len(MULTABENCH_CORE_IMAGE)}, expected {CORE_PER_MODALITY}"
assert len(MULTABENCH_CORE_TEXT) == CORE_PER_MODALITY, f"Core text half is {len(MULTABENCH_CORE_TEXT)}, expected {CORE_PER_MODALITY}"

_misplaced_image = [d.name for d in MULTABENCH_FULL_IMAGE if not is_image_dataset(d)]
_misplaced_text = [d.name for d in MULTABENCH_FULL_TEXT if not is_text_dataset(d)]
assert not _misplaced_image, f"Non-image datasets in the image tiers: {_misplaced_image}"
assert not _misplaced_text, f"Non-text datasets in the text tiers: {_misplaced_text}"

_uncurated_core = [d.name for d in MULTABENCH_CORE if not is_curated(d)]
assert not _uncurated_core, f"Core datasets must be uploaded MulTaBench datasets: {_uncurated_core}"

assert len(MULTABENCH_FULL_IMAGE) <= FULL_PER_MODALITY, f"Full image half exceeds {FULL_PER_MODALITY}"
assert len(MULTABENCH_FULL_TEXT) <= FULL_PER_MODALITY, f"Full text half exceeds {FULL_PER_MODALITY}"

assert set(PROMOTED_FROM) <= set(MULTABENCH_FULL), "PROMOTED_FROM keys must be Full members"
assert not set(PROMOTED_FROM.values()) & set(MULTABENCH_FULL), (
    f"Promoted-from ids must not also sit in a tier: "
    f"{[d.name for d in set(PROMOTED_FROM.values()) & set(MULTABENCH_FULL)]}"
)
