import os
from glob import glob
from os.path import basename, join

import kagglehub
import numpy as np
import pandas as pd

from multabench.benchmark.utils.constants import IMAGES_DIR
from multabench.benchmark.utils.curation import copy_images, save_dataset, task_type_from_name


DATASET_ID = "REG_IMAGE_SAO_PAULO_HOUSES"
SLUG_BASE = "multabench-full-sao-paulo-houses"
KAGGLE_SOURCE = "rogerio/image-based-property-price-prediction-in-so-paulo"

TARGET_COL = "price"
IMAGE_COL = "property pic"
IMAGE_SUBFOLDER = "Imagens"

# oldPrice is the same listing before a markdown. Condominio and IPTU are recurring charges set
# from the property value. The two Detalhes columns are comma-joined restatements of the amenity
# flags, location repeats bairro plus the zone, and title is the only real Portuguese prose.
COLS_TO_DROP = ["index", "title", "oldPrice", "Condomínio", "IPTU", "location",
                "Detalhes do imóvel", "Detalhes do condomínio"]

# Column names reach the encoder as "name: value", so the header is text the model reads.
RENAME = {
    "date": "listing date", "destaque": "featured", "Categoria": "property category",
    "Tipo": "property type", "Área útil": "floor area m2", "Quartos": "bedrooms",
    "Banheiros": "bathrooms", "Vagas na garagem": "parking spaces", "ZONA": "zone",
    "bairro": "neighbourhood", "Academia": "gym", "Elevador": "elevator",
    "Permitido animais": "pets allowed", "Piscina": "pool", "Portaria": "concierge",
    "Salão de festas": "party room", "Condomínio fechado": "gated community",
    "Segurança 24h": "24h security", "Portão eletrônico": "electric gate",
    "Área murada": "walled grounds", "Área de serviço": "laundry area",
    "Armários na cozinha": "kitchen cabinets", "Armários no quarto": "bedroom closets",
    "Churrasqueira": "barbecue", "Mobiliado": "furnished", "Quarto de serviço": "maid room",
    "Ar condicionado": "air conditioning", "Porteiro 24h": "24h doorman", "Varanda": "balcony",
}

# Values reach the encoder too, so the small closed vocabularies are translated with the headers.
_COUNT_COLS = ["Quartos", "Banheiros", "Vagas na garagem"]
_VALUES = {
    "Apartamentos": "apartment", "Casas": "house",
    "Padrão": "standard", "Casa de condomínio": "gated-community house", "Cobertura": "penthouse",
    "Casa de vila": "townhouse", "Loft": "loft", "Kitnet": "studio",
    "Duplex ou triplex": "duplex or triplex", "Sobrado": "two-storey house", "Flat": "flat",
    "OESTE": "west", "LESTE": "east", "SUL": "south", "NORTE": "north", "CENTRO": "central",
}
# A tenth of a percent of listings claim up to 608,608 m2, which is a data-entry slip rather than
# an estate; anything outside a plausible range becomes missing.
_AREA_RANGE = (10, 2000)


def _first_photo(dir_path: str) -> dict[int, str]:
    """One photo per listing, taken in filename order so the choice does not vary between runs.

    Photos are named <listing index>_<n>.png and spread over fourteen folders.
    """
    out: dict[int, str] = {}
    for path in sorted(glob(join(dir_path, IMAGE_SUBFOLDER, "*", "*.png"))):
        idx = int(basename(path).split("_")[0])
        out.setdefault(idx, join(*path.split(os.sep)[-3:]))
    return out


def _load_and_process(dir_path: str) -> pd.DataFrame:
    df = pd.read_csv(join(dir_path, "data.csv"), sep=";", low_memory=False)
    photos = _first_photo(dir_path)
    df[IMAGE_COL] = df["index"].map(photos)
    df = df[df[IMAGE_COL].notna()]
    df = df[df[TARGET_COL] > 0]
    # "5 ou mais" is the top bucket of an otherwise numeric count.
    for col in _COUNT_COLS:
        df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    area = pd.to_numeric(df["Área útil"], errors="coerce")
    df["Área útil"] = area.where(area.between(*_AREA_RANGE))
    for col in ["Categoria", "Tipo", "ZONA"]:
        df[col] = df[col].map(lambda v: _VALUES.get(v, v))
    df = df.drop(columns=COLS_TO_DROP, errors="ignore").rename(columns=RENAME)
    df = df.reset_index(drop=True)
    # Listings run to R$46M with a skew of 7.6, so R2 on the raw price would track whichever fold
    # caught a mansion.
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    assert df[IMAGE_COL].is_unique, "One photo per listing keeps the image split meaningful"
    return df


def curate(output_dir: str, slug: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dir_path = kagglehub.dataset_download(KAGGLE_SOURCE)
    df = _load_and_process(dir_path)
    print(f"  {len(df)} rows loaded")
    df = copy_images(df=df, image_col=IMAGE_COL, src_dir=join(dir_path, IMAGE_SUBFOLDER, ".."),
                     dst_dir=join(output_dir, IMAGES_DIR))
    save_dataset(df=df, output_dir=output_dir, target_col=TARGET_COL, dataset_id=DATASET_ID,
                 slug=slug, image_col=IMAGE_COL, task_type=task_type_from_name(DATASET_ID),
                 kaggle_source=KAGGLE_SOURCE)


if __name__ == "__main__":
    from multabench.benchmark.utils.curation import parse_curation_args
    args = parse_curation_args(SLUG_BASE, description=f"Curate {DATASET_ID} for MulTaBench-Full")
    curate(output_dir=args.output_dir, slug=args.slug)
