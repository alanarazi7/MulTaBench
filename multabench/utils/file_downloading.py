from http.client import InvalidURL
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from os import makedirs
from os.path import join, exists
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from pandas import DataFrame
import requests
from tqdm import tqdm

from multabench.constants import IMG_DEBUG
from multabench.benchmark.utils.curation import is_valid_curation_image


def download_url_image_column(df: DataFrame, img_folder: str, img_col: str) -> DataFrame:
    urls = df[img_col].tolist()

    # TODO: beware of parallelism, this might hit rate limits!
    with ThreadPoolExecutor(max_workers=4) as executor:
        df[img_col] = list(
            tqdm(
                executor.map(
                    lambda u: download_url_image(img_folder=img_folder, image_url=u),
                    urls
                ),
                total=len(urls),
                desc="Downloading images"
            )
        )
    images_before = len(df)
    images_now = len(df[df[img_col] != ""])
    if images_before != images_now:
        print(f"⚠️ Missing {images_before - images_now} images")
    return df

def download_url_image(img_folder: str, image_url: str) -> str | None:
    if image_url is None:
        return None
    if not isinstance(image_url, str):
        if IMG_DEBUG:
            print(f"🚫 Failed to download image from URL as it's from type {type(image_url)}, not string: {image_url}")
        return None
    url_filename = url_to_filename(image_url)
    img_path = join(img_folder, url_filename)
    if not exists(img_path):
        makedirs(img_folder, exist_ok=True)
        try:
            urlretrieve(url=image_url, filename=img_path)
        except (HTTPError, InvalidURL, URLError) as e:
            if IMG_DEBUG:
                print(f"🚫 Failed to download image from URL: {image_url}. Error: {e}")
            return None
        if not is_valid_curation_image(img_path):
            os.remove(img_path)
            return None
    return url_filename

def url_to_filename(u: str) -> str:
    return u.replace('/', '_').replace(':', '_')


def download_from_figshare(article_id: int, file_id: int, path: str):
    path = os.path.join(".tabstar_temp_files", f"{path}_{article_id}_{file_id}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_url = "https://api.figshare.com/v2"
    os.makedirs(path, exist_ok=True)
    files = requests.get(f"{base_url}/articles/{article_id}/files").json()
    f = next(x for x in files if x["id"] == file_id)
    download_url = f["download_url"]
    output_path = os.path.join(path, f["name"])
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
    return output_path



def unzip_url_dataset(src_path: str, dst_path: Optional[str] = None):
    if dst_path:
        os.makedirs(dst_path, exist_ok=True)
    else:
        dst_path = os.path.dirname(src_path)
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(dst_path)
