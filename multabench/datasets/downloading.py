import json
import os
from os.path import join, dirname
from time import sleep
from typing import Optional

import kagglehub
import openml
from openml.exceptions import OpenMLServerException
from pandas import read_csv, DataFrame

from multabench.datasets.all_datasets import KaggleDatasetID, OpenMLDatasetID, UrlDatasetID, MulTaBenchDatasetID, MultimodalDatasetID
from multabench.datasets.curation import MultimodalDataset, curate_dataset
from multabench.datasets.description import get_dataset_description
from multabench.datasets.kaggle_competitions import login_to_kaggle
from multabench.datasets.multimodal import MultimodalError, MultimodalState
from multabench.utils.io_handlers import load_txt
from multabench.utils.warnings import silence_kaggle_prints


ZIP_DATASETS: set[MultimodalDatasetID] = {
    UrlDatasetID.MUL_IMAGE_LEAGUE_OF_LEGENDS_SKIN_CATEGORY,
}

def download_multimodal_dataset(dataset_id: MultimodalDatasetID, for_annotation: bool = False) -> MultimodalDataset:
    return download_dataset(dataset_id=dataset_id, for_annotation=for_annotation)


def download_dataset(dataset_id: MultimodalDatasetID, multimodal_state: MultimodalState | None = None, for_annotation: bool = False, target_override: str | None = None) -> MultimodalDataset:
    if isinstance(dataset_id, MulTaBenchDatasetID):
        from multabench.benchmark.load import load_multabench_dataset
        from multabench.benchmark.utils.constants import MULTABENCH_KAGGLE_OWNER
        dataset = load_multabench_dataset(dataset_id, multimodal_state=multimodal_state)
        if for_annotation:
            url = f"https://www.kaggle.com/datasets/{MULTABENCH_KAGGLE_OWNER}/{dataset_id.value}"
            get_dataset_description(name=dataset_id.name, url=url, x=dataset.x, y=dataset.y)
            raise SystemExit
        return dataset
    elif dataset_id.name in {d.name for d in KaggleDatasetID}:
        with silence_kaggle_prints():
            return load_kaggle_dataset(dataset_id, multimodal_state=multimodal_state, for_annotation=for_annotation, target_override=target_override)
    elif dataset_id.name in {d.name for d in UrlDatasetID}:
        return load_url_dataset(dataset_id, multimodal_state=multimodal_state, for_annotation=for_annotation, target_override=target_override)
    elif dataset_id.name in {d.name for d in OpenMLDatasetID}:
        return load_openml_dataset(dataset_id, multimodal_state=multimodal_state, for_annotation=for_annotation, target_override=target_override)
    else:
        raise ValueError(f"Unsupported dataset ID type: {type(dataset_id)}")


def load_kaggle_dataset(dataset_id: KaggleDatasetID, for_annotation: bool = False, multimodal_state: MultimodalState | None = None, target_override: str | None = None) -> MultimodalDataset:
    dataset_value = dataset_id.value
    if dataset_value.count('/') == 1:
        dataset_value += '/'
    assert dataset_value.count('/') == 2
    dataset_name, file = dataset_value.rsplit('/', 1)
    # print(f"💾 Downloading Kaggle dataset {dataset_id.name}")
    for i in range(10):
        try:
            dir_path = None
            if not dataset_name.startswith('c/'):
                dir_path = kagglehub.dataset_download(dataset_name)
            df = _read_csv(path=os.path.join(dir_path, file), dataset_id=dataset_id) if file else None
            url = f"https://www.kaggle.com/{dataset_name}"
            dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id, dir_path=dir_path, multimodal_state=multimodal_state, target_override=target_override)
            if for_annotation:
                try:
                    description = _get_kaggle_dataset_description(dataset_name, dir_path=dir_path)
                except TypeError:
                    description = f"😭 NOT AVAILABLE!!!"
                get_dataset_description(name=dataset_value, url=url, x=dataset.x, y=dataset.y, desc=description)
                raise SystemExit
            return dataset
        except MultimodalError as e:
            raise MultimodalError(f"Kaggle dataset {dataset_id.name} is irrelevant: {e}")
        except Exception as e:
            raise ValueError(f"⚠️⚠️⚠️ Kaggle exception: {e}")
            #print(f": {e}")
            # sleep(120)
    raise ValueError("Failed to load Kaggle dataset")


def load_url_dataset(dataset_id: UrlDatasetID, for_annotation: bool = False, multimodal_state: MultimodalState | None = None, target_override: str | None = None) -> MultimodalDataset:
    for i in range(10):
        try:
            print(f"💾 Downloading URL dataset {dataset_id.name}")
            csv_path = get_csv_local_path(dataset_id=dataset_id)
            df = _read_csv(path=csv_path, dataset_id=dataset_id)
            dir_path = _get_dir_path(dataset_id)
            dataset = curate_dataset(x=df, y=None, dataset_id=dataset_id, dir_path=dir_path, multimodal_state=multimodal_state, target_override=target_override)
            if for_annotation:
                get_dataset_description(name=dataset_id.name, url=dataset_id.value, x=dataset.x, y=dataset.y)
                raise SystemExit
            return dataset
        except MultimodalError as e:
            raise MultimodalError(f"URL dataset {dataset_id.name} is irrelevant: {e}")
        except Exception as e:
            print(f"⚠️⚠️⚠️ URL Download exception: {e}")
            sleep(120)
    raise ValueError("Failed to load URL dataset")


def get_csv_local_path(dataset_id: MultimodalDatasetID) -> Optional[str]:
    if dataset_id in ZIP_DATASETS:
        return None
    dataset2local = {
        UrlDatasetID.REG_TEXT_CONSUMER_BABIES_R_US_PRICES: "babies_r_us.csv",
        UrlDatasetID.REG_TEXT_CONSUMER_BIKE_PRICE_BIKEWALE: "bikewale.csv",
        UrlDatasetID.REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER: "vancouber.csv",
        UrlDatasetID.REG_TEXT_PROFESSIONAL_ML_DS_AI_JOBS_SALARIES: "salaries.csv",
        UrlDatasetID.REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT: "scimagojr_2024.csv",
        UrlDatasetID.REG_TEXT_SOCIAL_BOOKS_GOODREADS: "goodreads.csv",
        UrlDatasetID.REG_TEXT_SOCIAL_MOVIES_ROTTEN_TOMATOES: "rotten_tomatoes.csv",
    }
    if dataset_id not in dataset2local:
        raise ValueError(f"Dataset ID {dataset_id} not recognized for URL datasets. We must copy-paste it locally.")
    local_filename = dataset2local[dataset_id]
    path = os.path.join(os.path.dirname(__file__), 'url_datasets', local_filename)
    return path



_SEMICOLON_DATASETS = {
    UrlDatasetID.REG_TEXT_PROFESSIONAL_SCIMAGOJR_ACADEMIC_IMPACT,
    UrlDatasetID.REG_TEXT_PROFESSIONAL_EMPLOYEE_RENUMERATION_VANCOUBER,
}

def _read_csv(path: Optional[str], dataset_id: MultimodalDatasetID) -> Optional[DataFrame]:
    if path is None:
        return None
    sep = ";" if dataset_id in _SEMICOLON_DATASETS else ","
    return read_csv(path, sep=sep)


def _get_dir_path(dataset_id: MultimodalDatasetID) -> Optional[str]:
    if dataset_id in ZIP_DATASETS:
        return join(f".tabstar_datasets_url_downloads", dataset_id.name)
    return None


def _get_kaggle_dataset_description(dataset_name: str, dir_path: str | None = None) -> str:
    api = login_to_kaggle()
    metadata_dir = join(dir_path, "downloaded-metadata")
    new_metadata_path = api.dataset_metadata(dataset_name, path=metadata_dir)
    data = load_txt(new_metadata_path)
    meta = json.loads(data)
    if isinstance(meta, str):
        # Weird kaggle behavior, it doesn't stores the data as a proper json but as a string
        meta = json.loads(meta)
    is_nested_dict = isinstance(meta, dict) and len(meta) == 1 and set(meta) == {"info"} and isinstance(meta["info"], dict)
    if is_nested_dict:
        meta = meta["info"]
    ret = ""
    for k in ["title", "subtitle", "keywords", "licenses", "description"]:
        if k in meta:
            ret += f"\n{k}: {meta[k]}"
    return ret


def load_openml_dataset(dataset_id: OpenMLDatasetID, for_annotation: bool = False, multimodal_state: MultimodalState | None = None, target_override: str | None = None) -> MultimodalDataset:
    for i in range(10):
        try:
            print(f"💾 Downloading OpenML dataset {dataset_id.name}")
            openml_dataset = openml.datasets.get_dataset(dataset_id.value, download_data=True, download_features_meta_data=True)
            x, y, _, _ = openml_dataset.get_data(target=openml_dataset.default_target_attribute)
            dataset = curate_dataset(x=x, y=y, dataset_id=dataset_id, multimodal_state=multimodal_state, target_override=target_override)
            if for_annotation:
                get_dataset_description(name=dataset_id.name, url=dataset_id.value, x=dataset.x, y=dataset.y)
                raise SystemExit
            return dataset
        except (OpenMLServerException, ConnectionError, FileNotFoundError) as e:
            print(f"⚠️⚠️⚠️ OpenML exception: {e}")
            sleep(120)
    raise ValueError("Failed to load OpenML dataset")
