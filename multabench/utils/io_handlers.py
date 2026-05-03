import json
import os
from zipfile import ZipFile


def load_json_lines(path: str) -> list[dict]:
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def load_txt(path: str) -> str:
    with open(path, 'r') as file:
        data = file.read()
    return data


def unzip_file(zip_path: str):
    assert zip_path.endswith('.zip')
    extract_to = zip_path.rstrip('.zip')
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)