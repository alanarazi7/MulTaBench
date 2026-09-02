"""MulTaBench tiering: Core (40) and Full (80).

MulTaBench-Core is the 40-dataset benchmark of the published paper (20 image-tabular + 20
text-tabular), frozen. MulTaBench-Full is the 80-dataset superset (40 + 40) promised in the
NeurIPS 2026 rebuttal; it grows dataset by dataset, so the invariants below are caps, never
floors.

The two tiers use *different* admission rules, both defined over the curation deltas in
`multabench.leaderboard.analysis.pass_matrix`:

    Core: Delta_Joint > delta AND Delta_Awareness > delta   (joint signal AND tabular awareness)
    Full: Delta_Joint > delta                               (joint signal only)

Both at delta=0.001 with a 3-of-5 curation-model committee. Relaxing Full to joint signal only
still excludes the two failure families the benchmark cares about -- pure-NLP tasks (the
unstructured modality alone is as good as the joint model) and redundant-text tasks (the
structured modality alone is) -- while admitting datasets where fine-tuning the encoder does not
additionally help.

Tier membership is deliberately typed over `MultimodalDatasetID`, not `MulTaBenchDatasetID`: a
Full member lives under its original source id (Kaggle/OpenML/URL) until it is re-hosted on
Kaggle as a curated `multabench-*` dataset. Use `is_curated()` / `pending_upload()` to tell the
two apart, and record the promotion in `PROMOTED_FROM` when it happens.

Core is kept as explicit literals rather than derived from the halves of `MulTaBenchDatasetID`:
that enum is "everything we ever uploaded" and will contain Full members too, so deriving Core
from it would silently absorb a 41st dataset into the published benchmark.
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


# The published 20 image-tabular datasets. Order is load-bearing (paper tables, datasets_summary.csv).
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

# The published 20 text-tabular datasets. Order is load-bearing (it fixes datasets_summary.csv rows).
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

# Full-only image-tabular datasets, on top of Core. Grows to 20 (camera-ready Track 1, item 3).
MULTABENCH_FULL_IMAGE_EXTRA: List[MultimodalDatasetID] = [
]

# Full-only text-tabular datasets, on top of Core. Admitted on Joint Signal alone (delta=0.001,
# 3-of-5 curation models) and ranked by (joint_pass_5, joint_pass_10, median Delta_Joint); see
# multabench/leaderboard/analysis/text_full_selection.py and text_full_selection.csv. These load
# through their original source until they are re-hosted as curated multabench-* datasets.
MULTABENCH_FULL_TEXT_EXTRA: List[MultimodalDatasetID] = [
    KaggleDatasetID.REG_TEXT_FOOD_WINE_POLISH_MARKET_PRICES,             # joint 5/5, 10/10 — CARTE
    KaggleDatasetID.REG_TEXT_SOCIAL_KOREAN_DRAMA,                        # joint 5/5, 10/10 — CARTE
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_SAUDI_ARABIA,       # joint 5/5, 10/10 — CARTE
    MulTaBenchDatasetID.MUL_TEXT_CONSUMER_COMPLAINT,                     # joint 5/5 uploaded — TextTabBench
    MulTaBenchDatasetID.MUL_TEXT_HEARTHSTONE_CARDS,                      # joint 5/5 uploaded — TextTabBench
    KaggleDatasetID.REG_TEXT_FOOD_WINE_VIVINO_SPAIN,                     # joint 5/5, 10/10 — CARTE
    KaggleDatasetID.REG_TEXT_FOOD_CHOCOLATE_BAR_RATINGS,                 # joint 5/5,  9/10 — CARTE
    KaggleDatasetID.REG_TEXT_SOCIAL_ANIME_PLANET_RATING,                 # joint 5/5,  9/10 — CARTE
    KaggleDatasetID.REG_TEXT_TRANSPORTATION_USED_CAR_PAKISTAN,           # joint 5/5,  9/10 — CARTE
    UrlDatasetID.REG_TEXT_SOCIAL_BOOKS_GOODREADS,                        # joint 5/5,  9/10 — Vectorizing
    KaggleDatasetID.REG_TEXT_FOOD_RAMEN_RATINGS_2022,                    # joint 5/5,  9/10 — CARTE+Vectorizing
    MulTaBenchDatasetID.BIN_TEXT_OSHA_INJURY,                            # joint 4/5 uploaded — TextTabBench
    OpenMLDatasetID.REG_TEXT_SPORTS_FIFA22_WAGES,                        # joint 5/5,  9/10 — CARTE
    OpenMLDatasetID.REG_TEXT_HOUSES_CALIFORNIA_PRICES_2020,              # joint 5/5,  9/10 — AutoMLMultimodal+TextTabBench
    KaggleDatasetID.REG_TEXT_FOOD_ALCOHOL_WIKILIQ_PRICES,                # joint 5/5,  8/10 — CARTE
    OpenMLDatasetID.REG_TEXT_CONSUMER_AMERICAN_EAGLE_PRICES,             # joint 5/5,  8/10 — AutoMLMultimodal
    KaggleDatasetID.REG_TEXT_HOUSES_AIRBNB_SEATTLE,                      # joint 5/5,  7/10 — TextTabBench
    MulTaBenchDatasetID.MUL_TEXT_NEWS_CHANNEL,                           # joint 5/5 uploaded — AutoMLMultimodal
    KaggleDatasetID.REG_TEXT_SOCIAL_MOVIES_DATASET_REVENUE,              # joint 4/5,  8/10 — CARTE
    UrlDatasetID.REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES,           # joint 4/5,  7/10 — CARTE
]

MULTABENCH_CORE: List[MultimodalDatasetID] = MULTABENCH_CORE_IMAGE + MULTABENCH_CORE_TEXT
MULTABENCH_FULL_IMAGE: List[MultimodalDatasetID] = MULTABENCH_CORE_IMAGE + MULTABENCH_FULL_IMAGE_EXTRA
MULTABENCH_FULL_TEXT: List[MultimodalDatasetID] = MULTABENCH_CORE_TEXT + MULTABENCH_FULL_TEXT_EXTRA
MULTABENCH_FULL: List[MultimodalDatasetID] = MULTABENCH_FULL_IMAGE + MULTABENCH_FULL_TEXT

# Curated MulTaBench id -> the original source id it was promoted from. The same dataset carries
# two different enum names in the two namespaces, so provenance cannot be recovered from names.
PROMOTED_FROM: Dict[MulTaBenchDatasetID, MultimodalDatasetID] = {
    MulTaBenchDatasetID.MUL_TEXT_CONSUMER_COMPLAINT: KaggleDatasetID.BIN_TEXT_FINANCIAL_CONSUMER_COMPLAINT,
    MulTaBenchDatasetID.MUL_TEXT_HEARTHSTONE_CARDS: KaggleDatasetID.MUL_TEXT_SOCIAL_HEARTHSTONE_CARD_GAME_WARCRAFT,
    MulTaBenchDatasetID.BIN_TEXT_OSHA_INJURY: KaggleDatasetID.BIN_TEXT_TRANSPORTATION_OSHA_ACCIDENT_INJURY_DATA,
    MulTaBenchDatasetID.MUL_TEXT_NEWS_CHANNEL: OpenMLDatasetID.MUL_TEXT_SOCIAL_NEWS_CHANNEL_CATEGORY,
    MulTaBenchDatasetID.BIN_TEXT_IMDB_GENRE: OpenMLDatasetID.BIN_TEXT_SOCIAL_IMDB_GENRE_PREDICTION,
    MulTaBenchDatasetID.MUL_TEXT_MELBOURNE_AIRBNB: OpenMLDatasetID.MUL_TEXT_HOUSES_MELBOURNE_AIRBNB,
}

# Uploaded and Joint-Signal-verified on their own artifacts, but not in the selected 20: both
# ranked below the cut on the pool evidence (IMDB Genre 4/5, Melbourne Airbnb 3/5) and scored
# better once re-hosted (5/5 and 4/5). Admitting them means displacing two pool-only members --
# a composition decision, so they wait here rather than pushing the text half past 40.
MULTABENCH_FULL_TEXT_UPLOADED_RESERVE: List[MulTaBenchDatasetID] = [
    MulTaBenchDatasetID.BIN_TEXT_IMDB_GENRE,        # joint 5/5 uploaded (was 4/5 in the pool)
    MulTaBenchDatasetID.MUL_TEXT_MELBOURNE_AIRBNB,  # joint 4/5 uploaded (was 3/5 in the pool)
]

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
    """Parse a CLI tier argument. Mirrors datasets.utils.dataset_from_name's friendly failure."""
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
    """True once the dataset is re-hosted on Kaggle as a curated `multabench-*` dataset."""
    return isinstance(dataset_id, MulTaBenchDatasetID)


def pending_upload(tier: "Tier | str" = Tier.FULL) -> List[MultimodalDatasetID]:
    """Tier members that still load through their original source instead of the unified API."""
    return [d for d in datasets_for_tier(tier) if not is_curated(d)]


def untiered_curated() -> List[MulTaBenchDatasetID]:
    """Uploaded datasets that belong to no tier -- a drift detector, not an error."""
    return [d for d in MulTaBenchDatasetID if get_tier(d) is None]


def _assert_no_duplicates(datasets: List[MultimodalDatasetID], label: str) -> None:
    seen = set()
    duplicated = [d.name for d in datasets if d in seen or seen.add(d)]
    assert not duplicated, f"{label} contains duplicate datasets: {duplicated}"


_assert_no_duplicates(MULTABENCH_CORE, "MULTABENCH_CORE")
_assert_no_duplicates(MULTABENCH_FULL, "MULTABENCH_FULL")

assert len(MULTABENCH_CORE_IMAGE) == CORE_PER_MODALITY, f"Core image half is {len(MULTABENCH_CORE_IMAGE)}, expected {CORE_PER_MODALITY}"
assert len(MULTABENCH_CORE_TEXT) == CORE_PER_MODALITY, f"Core text half is {len(MULTABENCH_CORE_TEXT)}, expected {CORE_PER_MODALITY}"

_misplaced_image = [d.name for d in MULTABENCH_FULL_IMAGE if not is_image_dataset(d)]
_misplaced_text = [d.name for d in MULTABENCH_FULL_TEXT if not is_text_dataset(d)]
assert not _misplaced_image, f"Non-image datasets in the image tiers: {_misplaced_image}"
assert not _misplaced_text, f"Non-text datasets in the text tiers: {_misplaced_text}"

_uncurated_core = [d.name for d in MULTABENCH_CORE if not is_curated(d)]
assert not _uncurated_core, f"Core datasets must be uploaded MulTaBench datasets: {_uncurated_core}"

assert set(MULTABENCH_CORE) <= set(MULTABENCH_FULL), "Core must be a subset of Full"

# Caps, never floors: Full is populated incrementally and must import cleanly while incomplete.
assert len(MULTABENCH_FULL_IMAGE) <= FULL_PER_MODALITY, f"Full image half exceeds {FULL_PER_MODALITY}"
assert len(MULTABENCH_FULL_TEXT) <= FULL_PER_MODALITY, f"Full text half exceeds {FULL_PER_MODALITY}"
assert len(MULTABENCH_FULL) <= 2 * FULL_PER_MODALITY, f"Full exceeds {2 * FULL_PER_MODALITY} datasets"

_aliased = [d.name for d in MULTABENCH_FULL if len([o for o in MULTABENCH_FULL if o.value == d.value]) > 1]
assert not _aliased, f"Full contains datasets sharing a source value: {_aliased}"

assert set(PROMOTED_FROM) <= set(MULTABENCH_FULL) | set(MULTABENCH_FULL_TEXT_UPLOADED_RESERVE), \
    "PROMOTED_FROM keys must be Full members or reserves"
assert not set(PROMOTED_FROM.values()) & set(MULTABENCH_FULL), (
    "PROMOTED_FROM values are pre-promotion ids and must not also sit in a tier: "
    f"{[d.name for d in set(PROMOTED_FROM.values()) & set(MULTABENCH_FULL)]}"
)
