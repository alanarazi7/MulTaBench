from os.path import join

from PIL import Image
from typing import Iterable

from pandas import Series

# Images are capped at MAX_IMAGE_SIZE (longest side) before being passed to DINO.
# DINO's processor resizes everything to 224px internally, so this cap has zero quality impact.
# Without it, high-res datasets (e.g. CBIS-DDSM mammography) load full-resolution images into RAM,
# causing ~90 GB memory usage per run and OOM crashes when multiple agents run on the same server.
# Capping at 512px reduces peak RAM to ~2 GB with no effect on the resulting embeddings.
MAX_IMAGE_SIZE = 512


def load_images(s: Series | Iterable, image_folder: str | None = None) -> list[Image.Image]:
    if isinstance(s, Series):
        s = s.tolist()
    return [load_image(v=v, image_folder=image_folder) for v in s]


def load_image(v: str | bytes, image_folder: str) -> Image.Image:
    if v is None:
        raise TypeError("🚨 Image path cannot be None!")
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if image_folder is not None:
        v = join(image_folder, v)
    img = Image.open(v)
    if img.mode == "P":
        img = img.convert("RGBA")
    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIZE:
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)
    return img
