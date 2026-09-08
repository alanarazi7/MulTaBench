import os
import re
import unicodedata
from os.path import exists, join
from typing import Optional

import kagglehub
import numpy as np
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_GOIAS_HOUSES"
SLUG_BASE = "multabench-full-goias-houses"
KAGGLE_SOURCE = "carloseduardogo/apartment-prices-in-goinia-gois-brazil"

TARGET_COL = "price"
IMAGE_COL = "department pic"
IMAGE_SUBFOLDER = join("imgs", "imgs")
NO_PRICE = "Sob consulta"
# condo_fee and iptu are recurring charges set from the property value.
# complete_description is Portuguese prose; the neighbourhood is lifted out of address first.
COLS_TO_DROP = ["id", "condo_fee", "iptu", "complete_description", "address"]

# Counts arrive as "3 quartos" or, for developments offering several unit types, "2 - 3 quartos".
_COUNT_COLS = ["numberOfRooms", "numberOfBathroomsTotal", "numberOfParkingSpaces", "floorLevel"]

_LISTING_STAGE = {"Destaque": "featured", "Super Destaque": "top featured",
                  "Em construcao": "under construction", "Pronto para morar": "ready to move in",
                  "Na planta": "off plan"}
_BUSINESS_TYPE = {"Venda": "sale", "A partir de": "from price", "Endereco": "address only"}


def _parse_brazilian_currency(price: str) -> Optional[float]:
    """'R$ 949.900' -> 949900.0. The dot is a thousands separator, so every dot is dropped; the
    source carries no decimal comma. The original recipe removed only one dot of two, scaling
    every price down by 1000."""
    price = str(price).replace("R$", "").strip()
    if price == NO_PRICE:
        return None
    return float(price.replace(".", ""))


def _parse_floorsize(floorsize: str) -> Optional[float]:
    floorsize = str(floorsize).replace(" m²", "").strip()
    try:
        return float(floorsize)
    except ValueError:
        return None


def _normalise(mapping: dict[str, str]) -> dict[str, str]:
    """Source labels carry accents that the mapping keys spell plainly."""
    strip = lambda t: "".join(c for c in unicodedata.normalize("NFD", t)
                              if unicodedata.category(c) != "Mn")
    return {k: v for key, v in mapping.items() for k in {key, strip(key)}}


def _parse_count(value: str) -> Optional[float]:
    """The midpoint of a range, so "2 - 3 quartos" orders between 2 and 3 rather than sorting as
    a string between "2 quartos" and "3 quartos"."""
    numbers = [float(n) for n in re.findall(r"\d+", str(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _neighbourhood(address: str) -> Optional[str]:
    """"Rua C238, 100 - Jardim America, Goiania - GO" -> "Jardim America". A place name, not
    prose: it is the strongest single predictor of price and survives the English-only bar."""
    parts = str(address).split(" - ")
    return parts[1].split(",")[0].strip() if len(parts) >= 2 else None


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "data.csv"), sep="|", low_memory=False)
    df[IMAGE_COL] = df["id"].apply(
        lambda i: f"{i}.png" if exists(join(dir_path, IMAGE_SUBFOLDER, f"{i}.png")) else None)
    df = df[df[IMAGE_COL].notna()]
    df[TARGET_COL] = df[TARGET_COL].apply(_parse_brazilian_currency)
    df["floorSize"] = df["floorSize"].apply(_parse_floorsize)
    df["neighbourhood"] = df["address"].apply(_neighbourhood)
    for col in _COUNT_COLS:
        df[col] = df[col].apply(_parse_count)
    df["tag_card"] = df["tag_card"].map(_normalise(_LISTING_STAGE))
    df["business_type"] = df["business_type"].map(_normalise(_BUSINESS_TYPE))
    # Every amenity is stored as its Portuguese label or as null, which is a boolean spelled as a
    # word. Amenities no listing claims carry no signal and are dropped outright.
    amenities = [c for c in df.columns
                 if df[c].dtype == object and df[c].nunique(dropna=True) <= 1
                 and c not in (IMAGE_COL, TARGET_COL)]
    for col in amenities:
        df[col] = df[col].notna().astype(int)
    df = df.drop(columns=[c for c in amenities if df[c].nunique() == 1], errors="ignore")
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    df = df[df[TARGET_COL].notna()].reset_index(drop=True)
    # Listings run R$25k to R$12M with a skew of 4.6, so R2 on the raw price would track whichever
    # fold caught a penthouse. log1p brings the skew to 0.57 and keeps the ordering intact.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
